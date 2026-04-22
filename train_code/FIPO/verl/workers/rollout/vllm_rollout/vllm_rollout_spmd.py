# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import getpass
import logging
import os
import pickle
import socket
import threading
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
from typing import Any

import numpy as np
import ray
import torch
import torch.distributed
import zmq
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][
        0
    ]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):
    def __init__(
        self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert (
            tensor_parallel_size <= torch.distributed.get_world_size()
        ), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(
                tensor_model_parallel_size=tensor_parallel_size
            )

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = (
                    model_hf_config.llm_config.max_position_embeddings
                )
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = (
                    model_hf_config.text_config.max_position_embeddings
                )
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            # print('max_position_embeddings')
            # print(max_position_embeddings)
            assert (
                max_position_embeddings >= config.prompt_length + config.response_length
            ), "model context length should be greater than total sequence length"
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(
            config.max_model_len or config.prompt_length + config.response_length
        )

        if (
            max_num_batched_tokens < max_model_len
            and self.config.enable_chunked_prefill
        ):
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = (
            "dummy" if config.load_format.startswith("dummy") else config.load_format
        )

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {
            key: val for key, val in engine_kwargs.items() if val is not None
        }
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        """
        详细定义了“批量生成（Batch Generation）”这一操作的数据接口规范。
        简单来说，它告诉开发者：“如果你传入一个批次的数据，我会返回什么样的数据结构，以及每个字段代表什么含义。”

        功能：接收一批提示词（Prompts），让模型生成对应的回复序列。
        输入 (batch)：类型是 DataProto（一个自定义的数据协议或类），包含了输入的文本或 Token。
        输出 (DataProto)：同样是一个 DataProto 对象，但里面填充了模型生成的详细数据。

        核心返回字段详解:

        prompts: [bsz, prompt_length]
            含义：原始的输入提示词 Token ID。
            形状：bsz (Batch Size, 批次大小) 行，prompt_length (提示词长度) 列。
            作用：保留输入副本，用于后续计算或日志记录。
        responses: [bsz, response_length]
            含义：模型生成的回复 Token ID。
            关键点：它不仅包含 LLM 生成的文本，还可能包含工具调用（tool_calls）的观察结果。
            形状：bsz 行，response_length (回复长度) 列。
        response_mask: [bsz, response_length]
            含义：一个掩码（Mask），用于区分哪些是模型自己写的，哪些是外部工具返回的。
            规则：
                1：代表这是 LLM 生成的 Token（模型需要对此负责，计算 Loss 时通常会关注这部分）。
                0：代表这是 观察到的 Token（如工具返回的结果）或 填充（Padding）。

        input_ids: [bsz, prompt_length + response_length]
            含义：完整的对话序列。
            构成：prompts + responses 的拼接。这是模型实际看到的完整上下文。
        attention_mask: [bsz, prompt_length + response_length]
            含义：注意力掩码。
            规则：1 代表有效 Token，0 代表 Padding。用于告诉模型在计算注意力时忽略填充部分。
        position_ids: [bsz, prompt_length + response_length]
            含义：位置 ID。
            作用：表示 Token 在序列中的增量位置，帮助模型理解顺序。

        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch[
            "position_ids"
        ]  # position_ids: 用于位置编码的张量，明确告诉模型每个 token 在序列中的绝对位置。

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)  #  获取当前批次的大小，即一次性处理多少个提示词。

        """
         处理原始 Prompt ID: 这部分代码处理非张量数据，目的是为后续步骤准备原始的、未经填充（padding）的 prompt ID 序列。
        """
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            """
            - non_tensor_batch: 这是一个字典，用于存储不适合放入张量（Tensor）的数据。
            - if "raw_prompt_ids" not in ...: 这是一个检查，防止重复计算。如果 non_tensor_batch 中还没有 raw_prompt_ids 这个键，才执行生成逻辑。
            - _pre_process_inputs(self.pad_token_id, idx[i]): 这是一个自定义函数（代码中未给出具体实现），
                - 它的作用很可能是从已经左填充的 idx[i] 序列中，移除填充部分，还原出该 prompt 原始的、紧凑的 token ID 列表。
            - np.array(..., dtype=object): 因为批次中每个 prompt 的原始长度可能不同，所以无法构成一个规整的二维数组。这里使用 dtype=object 创建一个 NumPy 数组，数组的每个元素都是一个长度可变的列表（即每个 prompt 的原始 token IDs）。
            """
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [
                    _pre_process_inputs(self.pad_token_id, idx[i])
                    for i in range(batch_size)
                ],
                dtype=object,
            )

        """
        这是一个安全检查，确保数据的一致性。
            - 它验证了从张量数据中获取的 batch_size 是否与刚刚生成的 raw_prompt_ids 列表的数量一致。
            - 如果不一致，说明在数据处理或分布式计算（sharding）的过程中出现了错误，导致批次内的样本数量对不上。
                此时代码会抛出一个 RuntimeError，阻止后续可能产生错误结果的计算。注释中提到的 "vllm sharding manager" 暗示这段代码可能运行在一个使用 vLLM 进行模型并行或数据并行的复杂环境中。
        """
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        """
        它根据输入数据是否包含多模态信息（如图像、音频等），将之前准备好的原始文本 ID 序列转换成 vLLM 能够识别和处理的字典列表格式。

        🧩 处理多模态数据分支
        它根据输入数据的类型（纯文本或多模态），灵活地构建出两种不同结构的输入列表 vllm_inputs，确保下游的 vLLM 推理引擎能够无缝处理不同类型的请求。

        当 non_tensor_batch 中包含 "multi_modal_data" 键时，说明当前批次的请求是多模态的（例如，图文问答）。
            1. if "multi_modal_data" in non_tensor_batch:: 检查是否存在多模态数据。
            2. zip(..., strict=True): - 使用 zip 函数将 raw_prompt_ids（原始文本ID列表）和 multi_modal_data（多模态数据列表）进行配对。
                                      - strict=True 参数确保了两个列表的长度必须完全相同，否则就会报错，这保证了每个文本 prompt 都能和一个多模态数据（如一张图片）正确对应。
            3. non_tensor_batch.pop(...): 从 non_tensor_batch 字典中取出并删除对应的键值对。使用 pop 是因为这些数据已经被提取出来用于构建新的格式，保留在原字典中已无必要。
            4. vllm_inputs.append({...}): 为每一对 (raw_prompt_ids, multi_modal_data) 创建一个符合 vLLM 格式的字典。
                这个字典明确包含了两个关键字段：
                - "prompt_token_ids": 文本部分的 token IDs。
                - "multi_modal_data": 对应的多模态数据，vLLM 会根据模型类型自动处理，例如将其识别为图像或音频。
        """
        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
                strict=True,
            ):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": multi_modal_data,
                    }
                )
        else:
            """
            处理纯文本分支
            """
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids}
                for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        """
        它的核心目的是确保输入数据的类型绝对安全且格式统一。vLLM 的底层接口通常期望 prompt_token_ids 是一个标准的 Python 列表（list），
        但之前的处理步骤（如 PyTorch 操作或 NumPy 处理）可能会留下张量（Tensor）或数组（Array）。这段代码就是为了消除这种格式差异。


        """
        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            """
            isinstance（...， list | np.ndarray）: 这里使用了 Python 的联合类型检查。它允许 prompt_token_ids 是 list（列表）或者 np.ndarray（NumPy 数组）。

            为什么要检查？
            在之前的步骤中（如 idx = prompts.batch["input_ids"]），数据很可能是 PyTorch Tensor 格式。虽然我们在之前的步骤中尝试将其转为 NumPy 数组，但为了代码的健壮性，这里必须防止“漏网之鱼”（比如直接传入了 Tensor 或其他奇怪的对象）。
            如果数据类型不对（比如传入了 PyTorch Tensor），直接传给 vLLM 可能会导致难以调试的底层错误。这里通过显式的 TypeError 提前拦截，让报错信息更清晰。
            """
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            """
            统一格式转换:
                强制转为 Python 原生列表：无论输入是 NumPy 数组还是原本的列表，这行代码都将其强制转换为 Python 原生的 list 类型。

            为什么要转为 List？
                - 序列化需求：vLLM 的客户端与后端引擎（EngineCore）通常通过 ZMQ 或 gRPC 进行通信。数据在网络传输前需要序列化（比如使用 msgpack 或 json）。
                            原生的 Python 列表比 NumPy 数组更容易、更标准地被序列化库处理。
                - 接口规范：vLLM 的 LLMEngine.add_request 或类似的 API 接口标准定义通常要求 prompt_token_ids 为 List[int]。
            """
            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        """
        根据当前的运行模式（推理模式 vs 验证模式），动态配置大模型的生成参数。

        它通过检查 prompts.meta_info 中的元数据，决定是采用“贪心搜索”（Greedy Search）还是“随机采样”（Sampling），并为这两种情况分别设置了严格的参数组合。

        do_sample: 决定是否开启随机采样。默认为 True（开启）。如果为 False，则意味着我们要进行确定性的生成。

        is_validate: 决定是否处于“验证/评估”阶段。默认为 False。在模型训练或微调过程中，通常需要在一个验证集上跑分，这时需要特定的参数设置。

        """
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            """
            当 do_sample 为 False 时，代码强制模型进入贪心解码模式。这是最确定的生成方式，模型每次都只选概率最高的那个词。
                - temperature: 0: 这是关键。温度设为 0 会消除所有随机性，使模型输出变得完全确定。
                - n: 1: 只生成 1 个结果。因为贪心搜索的结果是唯一的，生成多个也是重复的，所以设为 1 以节省资源。
                - top_p: 1.0 & top_k: -1: 这两个参数实际上被“禁用”了。
                - top_p=1.0 意味着考虑 100% 的累积概率（即所有词）。
                - top_k=-1 通常表示不限制候选词数量。
                - 配合 temperature=0，模型自然会选中概率最大的那个词，所以不需要额外的截断。
                - best_of: 1: 这是一个 vLLM 参数，表示只进行一次生成尝试（不进行多次生成选最优）。
            应用场景：通常用于需要标准答案的场景，比如代码生成、数学解题，或者在评估模型能力时排除随机性的干扰。
            """
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            """
            验证/评估模式:

            当 do_sample 为 True 且 is_validate 为 True 时，进入验证模式。
                - 使用配置参数: 这里没有写死数值，而是从 self.config.val_kwargs 中读取。
                  这说明在训练/验证的配置文件中，开发者已经预设好了一套“最佳参数”（比如 temperature=0.7, top_p=0.9）。
                - n: 1: 这里强制设为 1。注释解释了原因：if validate, already repeat in ray_trainer。
                  这意味着外层的训练框架（如 Ray Trainer）已经通过循环调用的方式实现了多次生成（比如为了计算 Pass@k 指标），
                  所以在单次请求内部不需要再让 vLLM 生成多次，否则会导致计算量指数级爆炸。
            """
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        """
        为当前批次的所有请求附加 LoRA 适配器信息。

        它通过一种“占位”策略，告诉推理引擎：“请对这个批次的所有样本，都应用我们已加载的第一个 LoRA 模型”。这在需要针对特定任务（如代码生成、特定角色对话）进行推理时非常关键。


        """
        lora_requests = None
        if self.lora_kwargs:
            """'

            if self.lora_kwargs:  如果 self.lora_kwargs：: 首先检查是否配置了 LoRA 相关的参数。
            如果 lora_kwargs 为空（None 或空字典），说明当前不需要使用 LoRA，直接跳过，lora_requests 保持为 None。


            list_loras()  list_loras（）: 这是一个 vLLM 引擎的方法，用于查询当前引擎中已经加载了哪些 LoRA 适配器。它返回的是 LoRA 的唯一标识符（ID）列表。
            lora_int_ids[0]: 代码这里做了一个非常实用的假设——“使用列表中的第一个”。
                    - 这通常意味着系统预设了只加载一个特定的 LoRA 用于当前任务，或者这是一个演示/测试环境，直接取第一个可用的即可。
                    - 在实际的多任务复杂场景中，这里可能会有更复杂的逻辑来选择特定的 lora_int_id。
            """
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                """
                构建 LoRARequest 对象：
                这是 vLLM 用来标识 LoRA 请求的数据结构。
                lora_int_id: 必须与引擎中加载的 ID 一致，引擎才知道你要用哪个权重。
                lora_path="/simon-stub-path": 这是一个非常有意思的“占位符”路径。
                    - 通常情况下，lora_path 指向磁盘上存放 LoRA 权重的文件夹。
                    - 但在这里，因为 LoRA 已经在内存中加载了（通过 list_loras() 可知），推理引擎其实不需要再去读磁盘。
                    - 传入一个假路径（Stub Path）只是为了满足 LoRARequest 构造函数的参数要求，避免报错，同时告诉引擎“别去读盘，直接用内存里那个 ID 对应的模型”。
                """
                lora_requests = [
                    LoRARequest(
                        lora_name=f"{lora_int_id}",
                        lora_int_id=lora_int_id,
                        lora_path="/simon-stub-path",
                    )
                ] * batch_size
                """
                    列表乘法 * batch_size：
                        - vLLM 的 generate 接口通常接受一个列表，列表长度必须等于 batch_size。
                        - 这意味着你需要为批次中的每一个样本单独指定一个 LoRA 配置。
                        - 因为我们要让当前批次的所有样本都使用同一个 LoRA，所以使用了 Python 的列表乘法，瞬间复制出 batch_size 个相同的 LoRARequest 对象。
                """

        """
        这段代码是整个推理流程的“执行与回收”阶段。
        它负责调用 vLLM 引擎真正生成文本，然后将 vLLM 返回的复杂对象解析、清洗、填充，最终转换成模型训练或后续处理所需的 PyTorch 张量格式。


        with self.update_sampling_params(**kwargs):: 这是一个上下文管理器。
            它的作用是临时修改生成参数。比如你可能在全局设置了默认参数，但在某次特定调用中（如验证阶段），你需要临时覆盖 temperature 或 top_p。
            进入 with 块时参数生效，退出后自动恢复，保证线程安全。
        
       
        """
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):

            """
            self.inference_engine.generate(...): 这是真正调用 vLLM 进行推理的地方。
               - prompts: 传入之前处理好的输入（token IDs）。
               - sampling_params: 采样策略（温度、Top-k 等）。
               - lora_request: 之前构建的 LoRA 适配器请求。
               - use_tqdm=False: 关闭进度条，通常因为在训练循环中打印进度条会刷屏，干扰日志。
            """
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            """
            解析 vLLM 输出:

            vLLM 返回的 outputs 是一个包含复杂对象的列表，这里将其“扁平化”并提取核心数据：

            双重循环:
                   1.外层 for output in outputs: 遍历批次中的每一个 Prompt。
                   2.内层 for sample_id ...: 遍历每个 Prompt 生成的多个结果（如果设置了 n > 1，即一次生成多个候选项）。
            
           
            """
            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    """
                    提取 Token IDs: output.outputs[sample_id].token_ids 获取生成的 token 序列，存入 response 列表。
                    """
                    response.append(response_ids)
                    """
                    提取对数概率 (Log Probs):
                    - 如果配置了 calculate_log_probs（通常用于强化学习 RLHF 阶段计算优势），代码会遍历每一步生成的概率分布。
                    - logprob[response_ids[i]].logprob: 这是一个关键查找。vLLM 返回的是整个词表的概率分布，这里只取出实际生成的那个 token 对应的概率值。  

                    """
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            """
            填充与对齐（关键步骤）:

            为什么要填充？ 
            vLLM 返回的是 Python 列表，且每个生成的序列长度可能不同（有的句子长，有的短）。但 PyTorch 的 Tensor 运算要求形状必须一致。

            pad_2d_list_to_length: 这是一个自定义工具函数。它将不规则的列表转换成一个规整的二维矩阵，不足长度的部分用 pad_token_id 补齐。
            .to(idx.device): 将数据从 CPU 内存移动到 GPU 显存（与输入 idx 相同的设备），为后续计算做准备。
            """
            response = pad_2d_list_to_length(
                response, self.pad_token_id, max_length=self.config.response_length
            ).to(idx.device)
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)
            """
            拼接最终序列:
            最终组装: 将输入部分 (idx, 即 prompt) 和生成部分 (response) 在序列维度上拼接起来。
            结果: seq 现在是一个完整的序列 [Prompt Tokens, Generated Tokens]，可以直接送入模型进行前向计算（例如计算 Loss）。
            """
            seq = torch.cat([idx, response], dim=-1)

        """
        为模型生成的回复部分（response）构建正确的位置编码（Position IDs）。

        在 Transformer 模型中，位置编码用于告诉模型每个 token 在序列中的绝对位置。
        既然 response 是接在 prompt 后面的，那么它的位置编号必须紧接着 prompt 的最后一个位置继续递增
        （例如，如果 prompt 结束于位置 10，response 就应该从位置 11 开始）。

        """
        """
        获取生成的 response 张量的序列长度（即生成了多少个 token）。假设 response 的形状是 (batch_size, response_length)
        """
        response_length = response.size(1)
        """
        生成相对位置序列:
        torch.arange(1, ...): 生成一个从 1 开始的序列 [1, 2, 3, ..., response_length]。
        为什么要从 1 开始？
            - 通常 position_ids 中，Prompt 的最后一个 token 的位置索引是 L-1（假设从 0 开始计数）。
            - 为了拼接，生成的第一个 token 应该是 Prompt 之后的第 1 个位置。
            - 后续代码（未展示）通常会将这个 delta_position_id 加上 prompt 的最后一个位置 ID，或者直接加上 prompt 的位置 ID 张量，从而得到绝对位置。这里的 1 代表了“相对于起始位置的偏移量”。

        """
        delta_position_id = torch.arange(
            1, response_length + 1, device=position_ids.device
        )

        """
        扩展批次维度:
        unsqueeze(0): 将形状从 (response_length,) 变为 (1, response_length)。这是为了引入批次维度。
        expand(batch_size, -1): 将张量在批次维度上复制 batch_size 份。
            - 此时，delta_position_id 的形状变为 (batch_size, response_length)。
            - 这意味着批次中的每一个样本都拥有相同的相对位置序列 [1, 2, ..., N]。
        """
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        """
        处理多模态/特殊架构（Qwen2VL MROPE）:

        这是一个非常具体的适配逻辑，专门针对像 Qwen2-VL 这样使用 MROPE 的模型。

        背景：普通的语言模型 position_ids 通常是 2D 的 (batch, seq_len)。但 Qwen2-VL 处理视频/图像时，
             使用了 3D 的位置编码（通常是 (batch, 3, seq_len)），分别代表时间、高度、宽度三个维度的位置信息。

        if position_ids.dim() == 3:: 检测输入的位置 ID 是否是 3 维的。如果是，说明当前处理的是多模态数据。

        view(batch_size, 1, -1): 先将 2D 张量变形为 (batch_size, 1, response_length)，在中间插入一个维度。

        expand(batch_size, 3, -1): 将中间的维度扩展为 3。
            - 这相当于把同一份位置序列 [1, 2, ..., N] 复制了三份，分别赋值给时间、高度和宽度维度。
            - 最终形状变为 (batch_size, 3, response_length)，以便能与 Qwen2VL 的输入进行拼接或相加。
        """
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                batch_size, 3, -1
            )

        """
        这段代码是整个推理流程的收尾与封装阶段。
          它负责计算生成内容的位置编码和注意力掩码，并将所有数据打包成一个标准的 DataProto 对象，以便返回给调用者（通常是训练循环或评估脚本）。

        1. 场景背景：
                - Prompt 是左填充（Left-pad）的，所以它的有效位置 ID 是从 0 开始递增的（例如 0, 1, 2, 3），填充部分是 0。
                - Response 是紧接着 Prompt 生成的，通常采用右填充（Right-pad）。
        2. 计算逻辑：
                - position_ids[..., -1:]: 取出 Prompt 的最后一个有效 token 的位置 ID。在注释的例子中，这个值是 3。
                - + delta_position_id: 将这个“基准位置”加上之前生成的“相对偏移量”（即 1, 2, 3...）。
                - 结果：3 + [1, 2, 3...] = [4, 5, 6...]。
                - 这确保了生成的回复在位置编码上是连续的，模型能正确理解回复是接在 Prompt 后面的。
        3. 拼接：最后将 Prompt 的 position_ids 和计算出的 response_position_ids 拼接起来，形成完整的序列位置编码。
        """
        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        """
        get_response_mask: 这是一个自定义函数，用于生成回复部分的掩码。
            - 它会标记出生成的 token 哪些是有效内容，哪些是填充（PAD）。
            - 通常还会处理 EOS (End of Sequence) token，确保序列结束符被正确标记。
        """
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        """
        打包数据:

        TensorDict: 这是一个方便管理批处理张量的数据结构（类似于字典，但专为 Tensor 设计）。
        数据汇总：
                prompts: 原始输入。
                responses: 模型生成的部分。
                input_ids: 拼接后的完整序列（用于计算 Loss）。
                attention_mask & position_ids: 对应的控制张量。
        注释含义：# all the tp ranks should contain the same data here 说明这是在使用张量并行的环境。
                这行代码确保所有 GPU 上的数据状态是一致的，因为接下来的计算（如前向传播）需要所有分片协同工作。

        """
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        """
        可选的 Log Probs：如果配置了计算概率，会将之前从 vLLM 提取的 rollout_log_probs 也放入批次中。这通常用于强化学习（RLHF）阶段，用来计算优势函数。

        
        """
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        """
        DataProto: 最终返回的封装对象，包含了所有张量数据（batch）和非张量数据（non_tensor_batch，如原始文本 ID 等）。
        """
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(
        self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs
    ):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()

        return address

    def _get_free_port(self):
        ip = ray.util.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address

    def init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

        _monkey_patch_compute_logits(
            self.inference_engine.worker.model_runner.model, len(self.tokenizer)
        )

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
