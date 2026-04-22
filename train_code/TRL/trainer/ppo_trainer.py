# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils import is_peft_available

from ..core import masked_mean, masked_whiten
from ..models import create_reference_model
from ..models.utils import unwrap_model_for_generation
from .ppo_config import PPOConfig
from .utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    generate_model_card,
    get_comet_experiment_url,
    get_reward,
    log_table_to_comet_experiment,
    peft_module_casting_to_bf16,
    prepare_deepspeed,
    print_rich_table,
    selective_log_softmax,
    truncate_response,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb


INVALID_LOGPROB = 1.0


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
        output = self.critic_backbone(**kwargs)
        logits = self.value_model.score(output.hidden_states[-1])
        return self.policy(**kwargs), logits


class PPOTrainer(Trainer):
    _tag_names = ["trl", "ppo"]

    def __init__(
        self,
        args: PPOConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        model: nn.Module,
        ref_model: Optional[nn.Module],
        reward_model: nn.Module,
        train_dataset: Dataset,
        value_model: Optional[nn.Module] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
        peft_config: Optional["PeftConfig"] = None,
    ) -> None:
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must make a copy of it, or `None` if you use peft."
            )

        self.args = args
        self.processing_class = processing_class
        self.policy_model = model

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        # Handle stop token settings: update policy model's generation_config to use provided stop token
        if args.stop_token and args.stop_token_id:
            raise ValueError("You cannot set both `stop_token` and `stop_token_id`.")
        elif args.stop_token:
            if args.stop_token == "eos":
                self.policy_model.generation_config.eos_token_id = self.stop_token_id = processing_class.eos_token_id
            else:
                raise ValueError(
                    f"Unknown `stop_token` {args.stop_token}. Allowed values are: `'eos'` and `None` (no stop token)."
                )
        else:
            self.policy_model.generation_config.eos_token_id = self.stop_token_id = args.stop_token_id  # None or int

        # peft support
        if not is_peft_available() and peft_config is not None:
            raise ImportError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_confg, we merge and unload it first
            if isinstance(self.policy_model, PeftModel):
                self.policy_model = self.policy_model.merge_and_unload()

            # get peft model with the given config
            self.policy_model = get_peft_model(self.policy_model, peft_config)
            if args.bf16 and getattr(self.policy_model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(self.policy_model)

        self.is_peft_model = is_peft_available() and isinstance(self.policy_model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(self.policy_model)

        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.value_model = value_model
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        if args.whiten_rewards:
            assert args.local_mini_batch_size >= 8, (
                f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
            )
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        for module in [self.policy_model, self.ref_model, self.value_model, self.reward_model]:
            if module is not None:
                disable_dropout_in_model(module)
        self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
        self.model.config = self.policy_model.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
            )

            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = self.ref_model.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model.policy).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.model_adapter_name or "default")

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        self.model = self.model.policy  # save only the policy

        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model

        super().save_model(output_dir, _internal_call)

        self.model = backup_model

        if self.is_deepspeed_enabled:
            self.deepspeed = backup_deepspeed

    def train(self):
        args = self.args #  PPOConfig
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_model
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device
        # while True 会创建一个无限循环，这意味着 repeat_generator 会不断地从 dataloader 中生成数据
        def repeat_generator():
            while True:
                yield from dataloader # yield from 是一个非常方便的语法，用于从另一个生成器中生成值。它会从 dataloader 中逐个生成值，直到 dataloader 被耗尽

        # 创建迭代器
        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)

        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)

        model.train()  # 设置模型为训练模式

        # trainer state initialization 
        # 初始化训练状态
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio 
        # 计算绝对值，用于日志，评估和保存
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1): # 遍历每个批次
            self.state.episode += 1 * args.batch_size
            # 每次从迭代器中取出一次数据：即从dataloader中加载一批数据
            data = next(iter_dataloader)
            with torch.no_grad(): # 使用推理模式，确保模型在推理过程中使用正确的设备
                queries = data["input_ids"].to(device)
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                values = []
                with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    # 生成query_responses和logitss数据
                    query_responses, logitss = batch_generation(
                        unwrapped_model.policy,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size): # 遍历每个批次
                    # step = local_rollout_forward_batch_size
                    query = queries[i : i + args.local_rollout_forward_batch_size] # 获取query: 输入的token ids
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]     
                    response = query_response[:, context_length:]
                    # 获取logits: 模型输出的logits
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax(logits, response)
                    del logits  # 删除logits，释放内存
                    torch.cuda.empty_cache()  # 清空缓存

                    if ref_policy is None:
                        with self.null_ref_context(): 
                            ref_output = forward(model.policy, query_response, processing_class.pad_token_id)
                    else: 
                        ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)

                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    # 计算参考模型的logits: 模型输出的logits
                    # 模型的输出 logits，即未经过 softmax 或其他激活函数处理的原始输出

                    # 对 ref_logits 进行温度缩放（temperature scaling），
                    # 这是一种常见的技术，用于调整模型输出的“锐度”或“平滑度”。温度缩放通常用于调整模型的置信度
                    # 温度参数，通常是一个正数，用于控制输出的平滑度。
                    # 温度值越低：输出越“尖锐”，即模型对某些类别的置信度越高
                    # 温度值越高：输出越“平滑”，即模型对所有类别的置信度更均匀
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    # 删除ref_output和ref_logits，释放内存  
                    del ref_output, ref_logits   
                    torch.cuda.empty_cache()

                    # ---Response Processing 1. truncate response after the first occurrence of `stop_token_id`---
                    # 截断响应，只保留第一个stop_token_id之后的部分
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )

                    # ---Response Processing 2. run reward model on the truncated responses---
                    # 将query和postprocessed_response连接起来
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    # 计算sequence_length: 第一个pad_token_id的索引     
                    # 获取sequence_length: 第一个pad_token_id的索引
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1
                    # 获取value_model: 价值模型
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model

                    # 计算价值模型: 价值模型    
                    full_value, _, _ = get_reward(
                        unwrapped_value_model, query_response, processing_class.pad_token_id, context_length
                    )
                    value = full_value[:, context_length - 1 : -1].squeeze(-1)

                    _, score, _ = get_reward(
                        reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    values.append(value)
                # 将所有列表中的张量拼接成一个张量
                # 将 responses 中的所有张量在第 0 维（即行方向）上拼接起来
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                values = torch.cat(values, 0)
                del (logprob, ref_logprob, full_value, value, score, unwrapped_model)
                torch.cuda.empty_cache()
                gc.collect()

                # ---Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id---
                # Completions not passing that filter will receive a lower score.   
                # 对生成的响应进行过滤，确保它们包含结束标记（eos_token_id），并对不包含该标记的响应施加惩罚
                # postprocessed_responses == self.processing_class.eos_token_id：创建一个布尔张量，表示哪些位置是结束标记。
                # torch.any(..., dim=-1)：按行检查是否有任何位置是结束标记，返回一个布尔张量，表示每行是否包含结束标记。
                contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
                # 对不包含结束标记的响应施加惩罚
                if self.args.missing_eos_penalty is not None:
                    # ~contain_eos_token：取反操作，表示哪些响应不包含结束标记。
                    # 对不包含结束标记的响应施加惩罚。
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                # 创建一个从 0 到 responses.shape[1] - 1 的序列，表示每个位置的索引。
                # .repeat(responses.shape[0], 1)：将该序列重复 responses.shape[0] 次，使其形状与 responses 一致。
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                # ：将 sequence_lengths 增加一个维度，使其形状为 (batch_size, 1)。
                # 创建一个布尔掩码，表示哪些位置超出了有效序列长度。
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                # 对对数概率进行掩码处理
                # 将 padding_mask 为 True 的位置替换为 INVALID_LOGPROB。
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                # 将序列长度加 1。
                sequence_lengths_p1 = sequence_lengths + 1
                # ze(1))：创建一个新的掩码，表示哪些位置超出了有效序列长度
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                # 将 padding_mask_p1 为 True 的位置替换为 0。
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # ---4. compute rewards 计算奖励---   
                # 计算 logprobs 和 ref_logprobs 之间的差异，这可以被视为 KL 散度的一个简化形式
                kl = logprobs - ref_logprobs  
                # 使用 args.kl_coef（KL 系数）乘以 KL 散度，并取负值
                non_score_reward = -args.kl_coef * kl
                # 使用 non_score_reward 初始化奖励张量
                rewards = non_score_reward.clone()
                # 创建一个从 0 到 rewards.size(0) - 1 的序列，表示每个样本的索引
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                # 使用 torch.where 条件选择，确定每个序列的实际结束位置。
                # 如果 sequence_lengths_p1 小于 rewards.size(1)，则使用 sequence_lengths_p1；否则使用 sequence_lengths。
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                rewards[[actual_start, actual_end]] += scores

                # ---5. whiten rewards--- 白化奖励
                if args.whiten_rewards:
                    # 使用 masked_whiten 函数对奖励进行标准化，同时使用掩码 ~padding_mask_p1 来指定哪些位置是有效的。
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    # 将超出有效长度的部分设置为 0
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # ---6. compute advantages and returns--- 计算优势和回报
                # 用于存储上一个时间步的 GAE 估计
                lastgaelam = 0
                # 用于存储每个时间步的优势，最终会反转以匹配时间顺序
                advantages_reversed = []
                # 生成的序列长度，即 responses 的第二维大小
                gen_length = responses.shape[1]
                # 计算 GAE
                for t in reversed(range(gen_length)):
                    # 如果当前时间步 t 不是最后一个时间步，则取 values[:, t + 1]；否则取 0.0。
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    # 计算 TD 误差
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    # 更新 GAE 估计
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    # 将当前时间步的优势添加到列表中
                    advantages_reversed.append(lastgaelam)
                # 使用 [::-1] 将列表反转，以匹配时间顺序
                # 使用 torch.stack 将列表中的张量堆叠成一个新的张量
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                # 回报是优势加上值函数的估计
                returns = advantages + values
                # 使用 masked_whiten 函数对优势进行标准化处理，同时使用掩码 ~padding_mask 来指定哪些位置是有效的。
                advantages = masked_whiten(advantages, ~padding_mask)
                # 使用 torch.masked_fill 将 padding_mask 为 True 的位置替换为 0
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                # 释放显存，避免内存不足的问题
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            # ppo训练循环
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                # 生成一个随机排列的数组
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                # 遍历local_mini_batch_size中的数据
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):

                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        # 前向传播和损失计算
                        with accelerator.accumulate(model):
                            # 当前微批次的结束索引
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            # 当前微批次的索引
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            # 当前微批次的优势、响应、查询响应、对数概率、回报、价值
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]
                            mb_return = returns[micro_batch_inds]
                            mb_values = values[micro_batch_inds]

                           # 模型的前向传播函数，返回模型的输出和价值预测
                            output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
                            # 获取logits
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            # 计算新的对数概率
                            new_logprobs = selective_log_softmax(logits, mb_responses)
                            # 超出有效长度的部分设置为无效值
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )

                            # 损失计算
                            vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                            vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                            # 剪切后的价值预测
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            # 计算价值损失的两种方式
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            # 取两种价值损失的最大值，以防止价值函数更新过大
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            # 计算掩码下的均值，确保只对有效位置进行计算
                            # ~padding_mask_p1[micro_batch_inds]：掩码，表示哪些位置是有效的
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                            # 计算价值损失的剪切比例
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                            )
                            # 新旧对数概率的差异
                            logprobs_diff = new_logprobs - mb_logprobs
                            # 概率比
                            ratio = torch.exp(logprobs_diff)

                            # 计算策略损失的两种方式
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            # 取两种损失的最大值:以防止策略更新过大
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            # 计算掩码下的均值，确保只对有效位置进行计算
                            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                            # 总损失，包括策略损失和价值损失
                            # args是PPOConfig
                            loss = pg_loss + args.vf_coef * vf_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            # 统计信息收集
                            # 收集各种统计信息，如近似 KL 散度、策略剪切比例、策略损失、价值损失、价值剪切比例、熵等。
                            with torch.no_grad():
                                pg_clipfrac = masked_mean(
                                    (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                                )
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    vf_clipfrac
                                )
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model.policy,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    _, score, _ = get_reward(
                        self.reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )
                    table["score"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent("""\
        @article{mziegler2019fine-tuning,
            title        = {{Fine-Tuning Language Models from Human Preferences}},
            author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
            year         = 2019,
            eprint       = {arXiv:1909.08593}
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="PPO",
            trainer_citation=citation,
            paper_title="Fine-Tuning Language Models from Human Preferences",
            paper_id="1909.08593",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
