# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import (
        index_first_axis,
        pad_input,
        rearrange,
        unpad_input,
    )
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import (
        index_first_axis,
        pad_input,
        rearrange,
        unpad_input,
    )


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get(
                "use_torch_compile", True
            )  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [
                        inputs[key] for inputs in micro_batch["multi_modal_inputs"]
                    ]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]],
                        dim=0,
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(
                    0, 1
                )  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(
                            rearrange(position_ids, "c b s ... -> (b s) c ..."), indices
                        )
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                        indices,
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import (
                        process_multi_modal_inputs_for_minicpmo,
                    )

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids,
                        attention_mask,
                        position_ids,
                        cu_seqlens,
                        multi_modal_inputs,
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(
                    input_ids_rmpad, shifts=-1, dims=1
                )  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = (
                            ulysses_pad_and_slice_inputs(
                                input_ids_rmpad,
                                position_ids_rmpad=position_ids_rmpad,
                                sp_size=self.ulysses_sequence_parallel_size,
                            )
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(
                    0
                )  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(
                                logits_rmpad
                            )  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[
                        :, -response_length - 1 : -1
                    ]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[
                    :, -response_length - 1 : -1
                ]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[
                        :, -response_length - 1 : -1
                    ]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[
                        :, -response_length - 1 : -1, :
                    ]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(
                                logits
                            )  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(
                                verl_F.entropy_from_logits, logits
                            )

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(
                max_norm=self.config.grad_clip
            )
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(
                self.actor_module.parameters(), max_norm=self.config.grad_clip
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module.parameters(), max_norm=self.config.grad_clip
            )

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(
                f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}"
            )
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(
        self, data: DataProto, calculate_entropy=False
    ) -> torch.Tensor:
        # """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        # Args:
        #     data (DataProto): a DataProto containing keys

        #         ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
        #         concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

        #         ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

        #         ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

        #         ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        # Returns:
        #     torch.Tensor: the log_prob tensor
        # """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = (
            ["multi_modal_inputs"] if has_multi_modal_inputs else []
        )

        data = data.select(
            batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys
        )

        if use_dynamic_bsz:
            max_token_len = (
                data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            )
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data, max_token_len=max_token_len
            )
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        """
        段代码是强化学习（特别是 PPO 算法）训练流程中的数据准备和预处理阶段。它的核心目的是从输入数据中提取必要的特征，
        根据配置筛选特定字段（如 KL 散度相关的数据），并最终将大数据集切分为小批次（Mini-batches），以便进行高效的迭代训练。

        这段代码对应了论文中“在收集的数据上进行多次小批量更新（multiple epochs of minibatch updates）”的核心思想

        self.actor_module.train():
            作用：将 PyTorch 模型设置为训练模式。
            细节：这会启用 Dropout 和 BatchNorm 等层的训练行为（在推理/评估时通常会用 eval() 关闭它们）。
        """
        self.actor_module.train()
        """
        temperature: 
         作用：从数据的元信息中提取采样温度。
         意义：温度参数控制模型生成文本的随机性。代码特意注释强调该参数必须存在，以防止因参数缺失导致的静默错误（Silent error）
        """
        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        """
        动态扩展：如果配置中启用了 use_kl_loss（KL 散度损失），则添加 ref_log_prob。
        """
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        """
        处理非张量数据（多模态支持）:
            作用：检查并处理非 Tensor 类型的数据（通常是复杂的 Python 对象，如图像、音频等）。
            逻辑：如果数据中包含 multi_modal_inputs（多模态输入，如图片特征），则将其加入待处理列表。这表明该框架支持多模态大模型（如 LLaVA, Qwen-VL）的 RL 训练。
        """
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = (
            ["multi_modal_inputs"] if has_multi_modal_inputs else []
        )

        """
        data.select(...):
            根据前面定义的 select_keys，从原始数据对象中筛选出需要的列，丢弃无关数据，减少内存占用。
        """
        data = data.select(
            batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys
        )

        """
        data.split(...):
            这是 PPO 论文中提到的“多次小批量更新”的具体实现。
            作用：将整个训练集（可能包含几千或几万个样本）切分为多个小批次（Mini-batches）。
            PPO 论文背景：论文中提到，PPO 的优势在于可以利用同一组采样数据进行多次（多个 Epochs）的小批量梯度更新。
                        这比传统的策略梯度方法（用完即弃）具有更高的样本效率。
            后续流程：代码的后续部分通常会遍历 mini_batches，对每一个小批次计算 Loss 并更新模型参数。
        """
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        """
        1.数据准备（前一步）：你采集了一批数据，并切分成了 mini_batches。
        2.外层循环（Epochs）：假设 ppo_epochs = 4。这意味着你要把这一批数据反复训练 4 遍。
        3.内层循环（Mini-batches）：在每一遍中，你依次取出一个小块数据。
        4.计算 Loss（核心）：
            - 利用当前模型计算新的概率 new_log_probs。
            - 计算比率 ratio= π_old/π_new计算 截断损失（Clipped Surrogate Loss）。
            - 更新参数：根据 Loss 更新一次模型参数。
        """
        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                """
                启用动态批次大小 (if self.config.use_dynamic_bsz:)
                    核心目的：最大化显存利用率，解决序列长度不一致导致的“由于 Padding 造成的计算浪费”问题。
                    逻辑解析：
                        背景：在自然语言处理中，样本长度差异很大（有的只有几个词，有的长达几千词）。
                             如果使用固定批次大小（例如固定 8 条数据），短的序列会被填充（Padding）到和最长序列一样长，导致大量无效计算。

                        max_token_len 计算：这里计算的是当前 GPU 能够处理的最大 Token 总数。
                            - ppo_max_token_len_per_gpu：单卡限制。
                            - ulysses_sequence_parallel_size：序列并行度。如果开启了 Ulysses 序列并行，序列被切分到多个 GPU 上，因此总的可处理长度是单卡限制乘以并行度。


                """
                if self.config.use_dynamic_bsz:
                    max_token_len = (
                        self.config.ppo_max_token_len_per_gpu
                        * self.ulysses_sequence_parallel_size
                    )
                    """
                    prepare_dynamic_batch：这是一个智能打包函数。它不再按“条数”切分，而是按“Token 总量”切分。
                            - 算法行为：它会尝试将尽可能多的样本塞进一个 micro_batch，只要这些样本的 Token 总数不超过 max_token_len。
                            - 效果：长序列会自动组成小批次（甚至单条），短序列会组成大批次。这就像玩“俄罗斯方块”，尽量把显存填满，不留空隙。
                    """
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch, max_token_len=max_token_len
                    )
                else:
                    """
                    分支二：固定批次大小:
                        核心目的：标准的梯度累积策略，用于在显存不足时模拟大 Batch Size 训练。
                        逻辑解析：
                                - gradient_accumulation (梯度累积步数)：
                                - 这是深度学习中的常用技巧。当显存放不下整个 mini_batch 时，我们将其拆分为更小的 micro_batch

                        作用：模型会依次处理这些 `micro_batch`，计算梯度并累加，直到处理完所有 `micro_batch` 后，
                             才执行一次 `optimizer.step()`。这样在数学上等价于一次性处理了整个 `mini_batch`
                    """
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    """
                    这是最朴素的切分方式。它简单粗暴地将 `mini_batch` 按照固定的样本数量（`ppo_micro_batch_size_per_gpu`）切成若干份。
                        缺点：如果切分后的样本长度差异大，依然会存在 Padding 浪费。
                    """
                    micro_batches = mini_batch.split(
                        self.config.ppo_micro_batch_size_per_gpu
                    )
                """
                梯度清零：确保新的计算不受旧数据的影响
                
                当你的 GPU 显存有限，无法承载一个很大的批次（Batch Size）时，你可以采用以下策略：
                    1.将一个大批次拆分成多个小批次（Micro-batches）。
                    2.依次对每个小批次进行前向和反向传播，但不清零梯度。这样，梯度就在多个小批次上被累加起来。
                    3.在处理完所有小批次后，累加的梯度就等效于在完整大批次上计算出的梯度。
                    4.此时，再调用 optimizer.step() 更新参数，并调用 optimizer.zero_grad() 清空梯度，准备下一轮。
                """
                self.actor_optimizer.zero_grad()
                '''
                它的主要任务是：遍历切分好的微批次（Micro-batches），从数据中提取关键张量，根据配置计算 Loss 的缩放系数（用于梯度累积），
                            并调用模型进行前向传播以获取当前的对数概率（Log Prob）和熵（Entropy）


                '''
                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    '''
                    model_inputs：将 Tensor 数据（batch）和非 Tensor 数据（non_tensor_batch，如多模态图像特征）合并，方便统一传入模型。

                    - response_mask：用于屏蔽 Prompt 部分和 Padding 部分。PPO 只对模型生成的“回答部分”计算 Loss，Prompt 部分不需要优化。
                    - old_log_prob：这是旧策略（生成这批数据时的模型版本）的对数概率。它是计算重要性采样比率（Importance Sampling Ratio）的分母，用于衡量新旧策略的差异。
                    - advantages：优势函数，衡量当前动作比平均水平好多少。
                    '''
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    '''
                    背景：在梯度累积中，我们将一个大 Batch 拆分成多个 Micro-batch。
                         为了保证数学上等价，我们需要将每个 Micro-batch 计算出的 Loss 进行缩放，使得累加后的梯度等于一次性处理大 Batch 的梯度。
                    '''
                    if self.config.use_dynamic_bsz:
                        '''
                        分支 1：动态批次 (use_dynamic_bsz=True)
                                场景：当序列长度差异大时，Micro-batch 的样本数（response_mask.shape[0]）是动态变化的（有的包得满，有的包不满）。
                                逻辑：缩放系数 = 当前样本数 / 目标总样本数。这意味着 Loss 是根据当前微批次在总批次中的占比来加权的。
                        '''
                        loss_scale_factor = (
                            response_mask.shape[0] / self.config.ppo_mini_batch_size
                        )
                    else:
                        '''
                        分支 2：固定批次 (use_dynamic_bsz=False)
                                场景：当序列长度差异不大时，Micro-batch 的样本数是固定的。
                                逻辑：缩放系数 = 1 / 梯度累积步数。这意味着 Loss 是按梯度累积步数来缩放的。
                        
                        '''
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    '''
                    calculate_entropy：
                        作用：熵（Entropy）用于鼓励探索，防止模型过早收敛到局部最优。
                        优化：计算熵需要额外的计算开销。如果配置中 entropy_coeff 为 0（即不使用熵正则化，参考材料 [2] 中提到 GRPO 有时会关闭熵），则跳过计算以节省显存和时间。
                    '''
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                        '''
                        _forward_micro_batch：
                                            核心动作：将数据输入当前的 Actor 模型。
                                            返回值：
                                                  log_prob：新策略的对数概率。这是计算 PPO Loss 的分子。
                                                  entropy：当前策略的熵值。
                        '''
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )

                    '''
                    作用：决定使用哪种具体的 PPO 变体算法来计算 Loss。
                        常见模式（参考材料 [1]）：
                                                vanilla：标准的 PPO 裁剪（Clip）损失。
                                                grpo_clip：GRPO 风格的裁剪，可能涉及 Group Relative 的优势计算。
                                                reinforce_with_baseline：带基线的 REINFORCE 算法。
                    '''
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    '''
                    FIPO实现：
                    它根据配置的 loss_mode 动态获取损失函数，并调用该函数来计算 PPO 的核心损失值以及一系列用于监控训练稳定性的详细指标

                    从注册表中获取对应的损失计算函数。
                    '''
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    if loss_mode == "future_kl":
                        (
                            pg_loss,
                            pg_clipfrac,
                            ppo_kl,
                            pg_clipfrac_lower,
                            influence_weights_mean,
                            influence_weights_min,
                            influence_weights_max,
                            total_clip_frac,
                            clip_frac_upper,
                            clip_frac_lower,
                            influence_weights_mean_raw,
                            raw_influence_weights_min,
                            raw_influence_weights_max,
                            neg_ratio_2_3,
                            neg_ratio_3_4,
                            neg_ratio_4_10,
                            neg_is_max,
                            neg_is_p995,
                            neg_is_p999,
                            neg_is_p75,
                            pos_is_max,
                            pos_is_median,
                            pos_is_p75,
                            pos_is_p995,
                            pos_is_p999,
                            pos_is_p25,
                            pos_is_min,
                            pos_mini_frac,
                            negative_approx_kl,
                        ) = policy_loss_fn(
                            old_log_prob=old_log_prob, # 旧策略概率（分母）
                            log_prob=log_prob,            # 新策略概率（分子）
                            advantage=advantages,       # 优势函数值（指导更新方向）
                            response_mask=response_mask, # 响应掩码 （只计算回答部分）
                            loss_agg_mode=loss_agg_mode, # 损失聚合模式 （如 mean, sum）              
                            config=self.config,   # 全局配置（包含 epsilon, KL 系数等超参）
                        )


                        '''
                        “数据记录与监控”环节:

                        它的核心作用是将上一阶段计算出的大量张量（Tensor）指标，转化为可以在日志系统（如 WandB, TensorBoard）中查看的标量（Scalar）或 CPU 数据。
                        这就像是飞机的“黑匣子”，记录了模型训练过程中的每一个细微状态，特别是关于重要性采样（Importance Sampling）和裁剪（Clipping）的详细统计。


                        '''
                        micro_batch_metrics["actor/influence_weights_mean"] = (
                            ''' 
                             数据清洗与转化:
                                    影响力权重（Influence Weights）：
                                        .detach()：将张量从计算图中分离出来，不再计算梯度。因为这是监控数据，不需要反向传播。
                                        .item()：将 PyTorch 张量转换为 Python 的原生数字（float）。这是为了节省显存，并让数据能被日志库（如 WandB）序列化

                            记录当前批次中，新旧策略比率的、均值、最小值、最大值。
                            '''
                            influence_weights_mean.detach().item()
                        )
                        # micro_batch_metrics["actor/influence_weights_std"] = influence_weights_std.detach().item()
                        micro_batch_metrics["actor/influence_weights_min"] = (
                            influence_weights_min.detach().item()
                        )
                        micro_batch_metrics["actor/influence_weights_max"] = (
                            influence_weights_max.detach().item()
                        )

                        '''
                        核心监控：裁剪统计（Clip Statistics）:

                        actor/IW_overall_clip_ratio (total_clip_frac)：
                            - 表示有多少比例的 Token 被裁剪了（即比率超出了 [1−ϵ,1+ϵ] 范围）。
                            - 意义：如果这个值很高（如 > 0.3），说明学习率可能太大，或者模型更新太激进，导致大量数据被“截断”而失去了梯度信息。
                        actor/IW_upper_clip_ratio (clip_frac_upper)：
                            - 撞上上界（1+ϵ ）的比例。通常发生在优势（Advantage）为正且很大时，模型想大幅增加某个动作的概率。
                        actor/IW_lower_clip_ratio (clip_frac_lower)：
                            - 撞上下界（1−ϵ ）的比例。通常发生在优势为负时，模型想大幅降低某个动作的概率。

                        '''
                        micro_batch_metrics["actor/IW_overall_clip_ratio"] = (
                            total_clip_frac.detach().item()
                        )
                        micro_batch_metrics["actor/IW_upper_clip_ratio"] = (
                            clip_frac_upper.detach().item()
                        )
                        micro_batch_metrics["actor/IW_lower_clip_ratio"] = (
                            clip_frac_lower.detach().item()
                        )

                        ''' 
                        原始权重监控（Raw Weights）
                            actor/influence_weights_mean_raw 等：
                                - 记录裁剪前的原始比率。
                                - 意义：通过对比“裁剪前”和“裁剪后”的数据，开发者可以知道 PPO 的裁剪机制到底“拦截”了多少疯狂的更新。如果原始比率波动巨大，说明模型处于极度不稳定的状态。

                        '''
                        # raw influence weight (before clip)
                        micro_batch_metrics["actor/influence_weights_mean_raw"] = (
                            influence_weights_mean_raw.detach().item()
                        )

                        micro_batch_metrics["actor/raw_influence_weights_min"] = (
                            raw_influence_weights_min.detach().item()
                        )
                        micro_batch_metrics["actor/raw_influence_weights_max"] = (
                            raw_influence_weights_max.detach().item()
                        )

                        '''
                        细粒度分布分析：正负样本分离
                            代码将样本分为正优势（Positive Advantage）和负优势（Negative Advantage）两组分别统计，这是非常高级的调试手段。

                        📉 负样本统计 (neg_ratio_..., neg_is_...)
                        对象：那些“坏动作”（Advantage < 0）。
                        neg_ratio_2_3, neg_ratio_3_4：统计比率在特定区间（如 2-3倍，3-4倍）的样本比例。
                        neg_is_p995, neg_is_p999：负样本比率的 99.5% 和 99.9% 分位数。
                        意义：这有助于发现长尾异常。例如，如果大部分负样本比率正常，但有 0.1% 的样本比率极高，这可能会导致梯度爆炸。

                        📈 正样本统计 (pos_is_...)
                        对象：那些“好动作”（Advantage > 0）。
                        pos_is_median：正样本比率的中位数。理想情况下，好动作的概率应该增加，所以这个值通常略大于 1。
                        pos_is_p25：25% 分位数。
                        意义：监控模型是否在正确地“鼓励”好动作，还是说连好动作都被过度裁剪了。

                        '''
                        # negative sample importance sampling ratio info
                        micro_batch_metrics["actor/neg_ratio_2_3"] = (
                            neg_ratio_2_3.detach().item()
                        )
                        micro_batch_metrics["actor/neg_ratio_3_4"] = (
                            neg_ratio_3_4.detach().item()
                        )
                        micro_batch_metrics["actor/neg_ratio_4_10"] = (
                            neg_ratio_4_10.detach().item()
                        )
                        # negative sample IS ratio basic stats
                        micro_batch_metrics["actor/neg_is_max"] = (
                            neg_is_max.detach().item()
                        )
                        micro_batch_metrics["actor/neg_is_p995"] = (
                            neg_is_p995.detach().item()
                        )
                        micro_batch_metrics["actor/neg_is_p999"] = (
                            neg_is_p999.detach().item()
                        )
                        micro_batch_metrics["actor/neg_is_p75"] = (
                            neg_is_p75.detach().item()
                        )
                        # postive sample IS ratio basic stats
                        micro_batch_metrics["actor/pos_is_max"] = (
                            pos_is_max.detach().item()
                        )
                        micro_batch_metrics["actor/pos_is_median"] = (
                            pos_is_median.detach().item()
                        )
                        micro_batch_metrics["actor/pos_is_p75"] = (
                            pos_is_p75.detach().item()
                        )
                        micro_batch_metrics["actor/pos_is_p995"] = (
                            pos_is_p995.detach().item()
                        )
                        micro_batch_metrics["actor/pos_is_p999"] = (
                            pos_is_p999.detach().item()
                        )
                        micro_batch_metrics["actor/pos_is_p25"] = (
                            pos_is_p25.detach().item()
                        )
                        micro_batch_metrics["actor/pos_is_min"] = (
                            pos_is_min.detach().item()
                        )
                        micro_batch_metrics["actor/pos_mini_frac"] = (
                            pos_mini_frac.detach().item()
                        )

                        '''
                        全局日志缓冲 (global_log_buffer):
                        作用：这里不仅记录了数字，还记录了原始数据。
                        negative_approx_kl：负样本的 KL 散度近似值。
                        responses：模型生成的文本 Token。
                        目的：这通常用于后续的定性分析。例如，当发现 KL 散度很高时，研究人员可以把这些 responses 还原成文本，人工查看模型到底生成了什么奇怪的内容导致了高 KL 值。
                        '''
                        # Collect data for global logging
                        micro_batch_metrics["actor/global_log_buffer"] = {
                            "negative_approx_kl": negative_approx_kl.detach().cpu(),
                            "responses": model_inputs["responses"].detach().cpu(),
                        }

                        # micro_batch_metrics["actor/valid_token_ratio_min"] = valid_token_ratio_min.detach().item()
                        # micro_batch_metrics["actor/valid_token_ratio_max"] = valid_token_ratio_max.detach().item()
                        # micro_batch_metrics["actor/valid_future_ratio_50"] = valid_future_ratio_50.detach().item()
                        # micro_batch_metrics["actor/valid_future_ratio_75"] = valid_future_ratio_75.detach().item()
                        # micro_batch_metrics["actor/valid_future_ratio_90"] = valid_future_ratio_90.detach().item()
                        # micro_batch_metrics["actor/valid_token_ratio_50"] = valid_token_ratio_50.detach().item()
                        # micro_batch_metrics["actor/valid_token_ratio_75"] = valid_token_ratio_75.detach().item()
                        # micro_batch_metrics["actor/valid_token_ratio_90"] = valid_token_ratio_90.detach().item()
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, pg_metrics = (
                            policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                            )
                        )
                        micro_batch_metrics.update(pg_metrics)

                    if entropy_coeff != 0:
                        '''
                        熵正则化（Entropy Regularization）逻辑。
                            简单来说，它的作用是通过调整损失函数，鼓励模型保持一定的“随机性”和“探索欲”，防止模型过早地变得“固执”或陷入局部最优
                        
                        高熵：模型输出概率分布很均匀（例如：选A、B、C的概率差不多）。这意味着模型还在探索，没有拿定主意。
                        低熵：模型输出概率集中在某一项（例如：选A的概率是99%）。这意味着模型非常确定，已经收敛了。

                        entropy_loss 计算：
                            - 利用 agg_loss 函数，根据掩码（response_mask）对熵值进行聚合（通常是求平均）。这确保了只计算生成部分的熵，忽略 Prompt 和 Padding。
                        
                        policy_loss = pg_loss - entropy_loss * entropy_coeff：
                            原理：我们的目标是最小化 policy_loss（Loss 越小越好）。
                                因为公式是 总Loss = PG_Loss - 熵，为了减小总 Loss，优化器会倾向于增大熵。
                            效果：这就像在奖励模型“保持困惑”。如果模型太早确定某个词是唯一的正确答案（熵变低），这一项 Loss 就会变大（因为减去一个很小的熵，总 Loss 相对变大），从而“惩罚”这种过早确定的行为。
                        
                        这段代码通过数学手段实现了“探索与利用”的平衡：
                            - pg_loss 负责利用：让模型做对的事情（拿高分）。
                            - - entropy_loss 负责探索：让模型不要只做那一件事（保持多样性）。
                        '''
                        entropy_loss = agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    '''
                    KL 散度惩罚（KL Divergence Penalty）:
                        如果说之前的“熵”是为了鼓励探索，那么这里的 KL 散度就是为了防止模型“学坏了”或者“忘本”。
                        它充当了一个“锚点”，确保正在训练的模型不会偏离原始参考模型太远。

                        风险：如果只追求奖励，模型可能会钻空子。例如，为了拿高分，模型可能会生成重复的乱码、奇怪的句式，或者完全忘记了人类语言的语法规范。这被称为奖励黑客（Reward Hacking）。
                        解决：我们需要一个参考模型（Reference Model）（通常是初始的 SFT 模型），它代表了“正常的、像人类的语言分布”。
                        KL 散度的作用：计算当前模型（Actor）和参考模型（Ref）之间的差异。差异越大，惩罚越大。这迫使模型在“拿高分”和“像人话”之间寻找平衡。
                    '''
                    if self.config.use_kl_loss:
                        '''
                        这里取出了参考模型对当前生成内容的对数概率。注意，参考模型在训练过程中是冻结（Frozen）的，它不参与更新，只是作为一个静态的标尺
                        '''
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        '''
                        输入：比较“新策略”和“旧参考策略”在每一个 Token 上的概率分布差异。
                        逻辑：如果 Actor 模型在这个 Token 上的概率分布与 Ref 模型差异很大（例如 Ref 觉得“你好”概率高，Actor 觉得“嘿嘿”概率高），kld 的值就会变大。

                       
                        '''
                        kld = kl_penalty(
                            logprob=log_prob,           # 当前 Actor 模型的对数概率
                            ref_logprob=ref_log_prob,   # 参考模型的对数概率
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld,
                            loss_mask=response_mask, # 同样只对生成的回答部分（response_mask）计算 KL，忽略 Prompt。
                            loss_agg_mode=loss_agg_mode,
                        )

                        '''
                         总Loss = PG_Loss + KL_Loss，为了减小总 Loss，优化器必须减小 KL Loss
                        '''
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

                        '''
                        actor/kl_loss：记录当前的 KL 惩罚值。
                            调试意义：这是 RLHF 训练中最重要的监控指标之一。
                                如果 KL 值接近 0：说明模型几乎没动，或者参考模型和当前模型一模一样。
                                如果 KL 值非常大：说明模型已经发生了剧烈变化，可能已经“崩坏”或者正在学习非常激进的新策略。通常我们会设定一个目标 KL（如 0.02），如果超过这个值，说明惩罚力度可能不够。
                        '''
                        micro_batch_metrics["actor/kl_loss"] = (
                            kl_loss.detach().item() * loss_scale_factor
                        )
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    '''
                    损失缩放与反向传播:
                        为什么要乘以 loss_scale_factor？
                            无论是动态批次还是固定批次，这里的核心目的都是为了正确地进行梯度累积。
                            背景：在梯度累积中，我们将一个大批次（Mini-batch）拆分成多个小批次（Micro-batches）。为了保证数学上等价，我们需要确保所有 Micro-batch 累加后的梯度，等于一次性处理整个 Mini-batch 的梯度。
                            缩放逻辑：
                                固定批次：loss_scale_factor 通常是 1 / gradient_accumulation_steps。这意味着每个 Micro-batch 的 Loss 只贡献总梯度的 1/N。
                                动态批次：loss_scale_factor 是 当前MicroBatch样本数 / 目标MiniBatch样本数。因为每个 Micro-batch 的大小可能不同（有的包得满，有的包不满），所以必须按比例加权。
                            结果：通过乘以这个系数，我们确保了无论 Micro-batch 大小如何，累加后的梯度方向都是正确的。
                    '''
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    '''
                    指标收集与监控: 这段代码将当前 Micro-batch 的关键指标存入字典，以便后续记录到日志系统（如 WandB/TensorBoard）


                    actor/pg_loss：
                        策略梯度损失（Policy Gradient Loss）。这是 PPO 的核心目标函数，反映了模型在“拿奖励”方面的表现。
                        注意：这里也乘以了 loss_scale_factor，是为了记录对总梯度有实际贡献的 Loss 值，而不是原始的局部 Loss。
                    actor/pg_clipfrac：
                        裁剪分数（Clipping Fraction）。表示有多少比例的 Token 被 PPO 的裁剪机制限制了更新。
                        意义：这是判断 PPO 训练是否稳定的最重要指标。如果该值过高（如 > 0.3），说明模型更新幅度过大，被强行截断，训练可能不稳定。
                    actor/ppo_kl：
                        KL 散度近似值。衡量当前策略与旧策略（或参考策略）的差异。
                        意义：用于监控模型是否偏离太远。
                    actor/pg_clipfrac_lower：
                        下界裁剪分数。特指因为比率过小（低于 1−ϵ ）而被截断的比例。这有助于分析模型在“抑制坏动作”时的行为。

                    '''
                    '''
                    detach():
                        pg_loss, pg_clipfrac, ppo_kl 等变量都是从模型的输出（log_prob）计算得来的，因此它们天然地与模型参数相连，是计算图的一部分。
                        如果不使用 .detach()，当你把这些指标存入 micro_batch_metrics 字典时，Python 会持有这些张量的引用。这会导致一个严重的问题：整个计算图无法被释放，会一直驻留在显存（GPU内存）中。
                    '''
                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item()
                            * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    '''
                    将当前 Micro-batch 的指标合并到整个 Mini-batch 的指标字典中。
                    后续：通常在这个循环结束后，会对这些指标求平均值，从而得到这一步（Step）的最终训练报告
                    '''
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
