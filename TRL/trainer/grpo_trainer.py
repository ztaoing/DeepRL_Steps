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

import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch

import torch
import torch.utils.data
import transformers
from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_model,
    set_seed,
)
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from ..data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from ..extras.profiling import profiling_decorator
from ..import_utils import is_rich_available, is_vllm_available
from ..models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from .callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
from .utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

'''
RepeatRandomSampler 的类，它是一个自定义的 PyTorch 数据采样器。
这个采样器的目的是以一种结构化的方式重复数据集的索引，适用于需要对数据进行多次重复采样的场景。
'''
class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

        data_source：数据集，必须是一个可测量长度的对象（如 torch.utils.data.Dataset）。
        mini_repeat_count：每个索引在每个批次中重复的次数。
        batch_size：每个批次中独特的索引数量，默认为 1。
        repeat_count：整个采样过程重复的次数，默认为 1。
        seed：随机种子，用于确保采样的可重复性，默认为 None

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator 使用 torch.Generator 创建了一个本地随机数生成器，如果提供了种子，则使用 manual_seed 方法设置种子。
        if seed is not None:
            self.generator.manual_seed(seed)

    '''
    torch.randperm：生成一个随机排列的索引列表。
    分块操作：将索引列表按 batch_size 分块。
    过滤：丢弃那些长度不等于 batch_size 的块，确保每个批次的大小一致。
    重复生成：按照 repeat_count 和 mini_repeat_count 的要求，重复生成每个索引。
    '''

    def __iter__(self):
        # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [
            indexes[i : i + self.batch_size]
            for i in range(0, len(indexes), self.batch_size)
        ]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index
    # 计算并返回采样器的总长度，即数据集大小乘以重复次数和每个索引的重复次数。
    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count

# 这个类继承自 Hugging Face 的 Trainer 类，并扩展了其功能以支持 GRPO 算
class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).
    
    这段代码提供了一个使用 GRPOTrainer 的示例
    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel], # model：可以是一个字符串（模型 ID）或一个 PreTrainedModel 对象。如果是一个字符串，将使用 AutoModelForCausalLM.from_pretrained 加载模型
        reward_funcs: Union[RewardFunc, list[RewardFunc]], # reward_funcs：可以是一个或多个奖励函数。每个奖励函数可以是一个字符串（模型 ID）、一个 PreTrainedModel 对象或一个自定义的 Python 函数。
        args: Optional[GRPOConfig] = None, # GRPOConfig 对象，包含训练的配置参数，如学习率、批量大小、生成长度等。
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None, # 处理类（PreTrainedTokenizerBase），用于对输入数据进行预处理。
        reward_processing_classes: Optional[ # reward_processing_classes：与奖励函数对应的处理类（PreTrainedTokenizerBase），用于对奖励函数的输入进行预处理。
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None, # 训练过程中的回调函数列表。
        optimizers: tuple[ # 优化器和学习率调度器的元组
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str): # 检查 model 是否是一个字符串。如果是字符串，表示传入的是模型的名称或路径。
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if (
                isinstance(torch_dtype, torch.dtype)
                or torch_dtype == "auto"
                or torch_dtype is None
            ): # 验证 torch_dtype
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype) # 从 torch 模块中获取对应的 torch.dtype
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False
                if args.gradient_checkpointing
                else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError(
                    "PEFT is required to use `peft_config`. Run `pip install peft`."
                )
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing: # 检查是否启用了梯度检查点
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_id, **model_init_kwargs
            )
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model.config._name_or_path, padding_side="left"
            )

        # 奖励函数 Reward functions
        if not isinstance(reward_funcs, list): # 检查 reward_funcs 是否是一个列表。如果不是列表，将其转换为包含单个元素的列表。这确保了后续的处理逻辑可以统一处理单个奖励函数和多个奖励函数的情况。
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs): # 遍历 reward_funcs 列表，同时获取每个奖励函数的索引 i 和值 reward_func。
            if isinstance(reward_func, str): # 检查当前的奖励函数是否是一个字符串。如果是字符串，表示它是一个模型 ID。
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs  
                )
        # 奖励函数
        self.reward_funcs = reward_funcs 

        # 奖励权重 Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # 奖励处理类 Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs) 
        ):
            if isinstance(reward_func, PreTrainedModel): 
                if reward_processing_class is None: 
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class. 
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class 
        self.reward_processing_classes = reward_processing_classes

        # 数据整理 Data collator
        def data_collator(features):  # No data collation is needed in GRPO 
            return features

        # 训练参数 Training arguments
        self.max_prompt_length = args.max_prompt_length # 提示的最大长度
        self.max_completion_length = ( 
            args.max_completion_length
        )  # = |o_i| in the GRPO paper 是GRPO论文中的o_i
        self.num_generations = args.num_generations  # = G in the GRPO paper  是GRPO论文中的G
        self.use_vllm = args.use_vllm

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon = args.epsilon # 是GRPO论文中的epsilon
        # 跟踪迭代的次数Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle. 
        self._step = 0 
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps # 缓冲区，用于在多个更新中重用生成的输出

        '''
        The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the 
        "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        This acts as a flag to indicate that the warning has already been issued. 
        '''
        model.warnings_issued["estimate_tokens"] = True 

        # 初始化指标 Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.log_completions = args.log_completions

        super().__init__( 
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [ # 用于检查是否可以被整除
            n_gen for n_gen in range(2, global_batch_size + 1)  if (global_batch_size) % n_gen == 0 
        ]
        if self.num_generations not in possible_values: 
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen
                for n_gen in range(2, global_batch_size + 1)
                if (global_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with 
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm: 
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if (
                    vllm_device.split(":")[0] == "cuda"
                    and int(vllm_device.split(":")[1]) >= torch.cuda.device_count()
                ):
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {
                    f"cuda:{idx}" for idx in range(self.accelerator.num_processes)
                }:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch(
                    "torch.distributed.get_world_size", return_value=1
                )
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                    return_value=None,
                )
                with world_size_patch, profiling_patch: # 使用上下文管理器，确保在退出上下文管理器时，world_size_patch和profiling_patch被正确的清理
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=self.args.vllm_enable_prefix_caching,
                        max_model_len=self.args.vllm_max_model_len,
                    )

                # Guided decoding, if enabled
                if args.vllm_guided_decoding_regex is not None:
                    guided_decoding = GuidedDecodingParams(
                        backend="outlines", regex=args.vllm_guided_decoding_regex
                    )
                else:
                    guided_decoding = None

                # 采样参数 Sampling parameters
                self.sampling_params = SamplingParams( 
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                    n=args.num_generations,
                )

            self._last_loaded_step = (
                0  # tag to avoid useless loading during grad accumulation 用于避免在梯度累积期间加载不必要的累积权重
            )

            # 当使用 vLLM 时，主进程负责加载模型权重。这可能导致进程不同步，并且似乎会导致 DeepSpeed 在初始化期间挂起。为了防止这种情况，我们在 vLLM 完全初始化后同步所有进程。
            # When using vLLM, the main process is responsible for loading the model weights. This can cause process 
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we 
            # synchronize all processes after vLLM has been fully initialized. 
            self.accelerator.wait_for_everyone() # 等待所有进程完成初始化
        else:
            self.generation_config = GenerationConfig( # 生成配置
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the 
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        # 梯度累积需要缩放的损失。通常，父类中的损失缩放取决于模型是否接受损失相关的 kwargs。由于我们计算自己的损失，这个检查是无关的。我们设置 self.model_accepts_loss_kwargs 为 False 以启用缩放。
        self.model_accepts_loss_kwargs = False

        # 添加标签到模型 Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        if args.sync_ref_model: # 如果设置为 True，则添加回调函数
            self.add_callback(
                SyncRefModelCallback(
                    ref_model=self.ref_model, accelerator=self.accelerator # 引用模型和加速器
                )
            )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True
                )

    def _set_signature_columns_if_needed(self): # 如果设置为 True，则删除未使用的列 
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler: # 获取训练采样器
        # Returns a sampler that 
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generaations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                     |     GPU 0     |     GPU 1     |     GPU 2    | 
        #
        #               global_step   step     <───────>  num_generations=3
        #                                      <───────────> per_device_train_batch_size=4
        #                ▲   0          0      0   0   0   1   1   1   2   2   2   3   3   3  │ 第一次迭代
        #  grad_accum=3  │   0          1      4   4   4   5   5   5   6   6   6   7   7   7  │ Generate completions for each prompt 为每个提示生成完成
        #                ▼   0          2      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    1          3      0   0   0   1   1   1   2   2   2   3   3   3  │ The sampled prompts are the same as in the first iteration 采样的提示与第一次迭代中的提示相同
        #                    1          4      4   4   4   5   5   5   6   6   6   7   7   7  │ Reuse the completions (here, once, because num_iterations=2)重用完成（这里，一次，因为 num_iterations=2）
        #                    1          5      8   8   8   9   9   9  10  10  10  11  11  11  │ 
        #
        #                    2          6     12  12  12  13  13  13  14  14  14  15  15  15  
        #                    2          7     16  16  16  17  17  17  18  18  18  19  19  19  第三次迭代
        #                    2          8     20  20  20  21  21  21  22  22  22  23  23  23
        #                                          ...
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler: # 获取评估采样器
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    # 减少50-80%的GPU内存使用 ；增加约20-30%的训练时间（需要重新计算激活值）
    def _enable_gradient_checkpointing( # 启用梯度检查点  
        self, model: PreTrainedModel, args: GRPOConfig 
    ) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled 确保 use_cache 被禁用
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT 
        if is_peft_model(model): # 检查模型是否是 PEFT 模型
            model.base_model.gradient_checkpointing_enable() 
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs
            or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model 

    # Get the per-token log probabilities for the completions for the model and the reference model 获取每个token的对数概率
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep): # 获取每个token的对数概率
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded 因为我们稍后会排除序列的最后一个logits
        logits = model( # 调用模型获取logits，input_ids是输入的token ids，attention_mask是注意力掩码，logits_to_keep是保留的logits数量
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[
            :, :-1, :
        ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred 排除最后一个logits，因为它对应于下一个token的预测

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves. 
        # 对于 transformers<=4.48，logits_to_keep 参数不受支持，所以这里我们自己删除logits。
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(
            logits, input_ids
        )  #  compute logprobs for the input tokens 计算输入token的对数概率

    @profiling_decorator
    def _move_model_to_vllm(self): # 将模型移动到vLLM
        with unwrap_model_for_generation(
            self.model,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):   # 检查模型是否是编译模块
                unwrapped_model = unwrapped_model._orig_mod 
            if is_peft_model(unwrapped_model):  # 检查模型是否是 PEFT 模型
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                # Remove base_model and base_layer prefixes 删除 base_model 和 base_layer 前缀
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v
                    for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora") 删除带有适配器前缀的值（例如："_lora"）
                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if unwrapped_model.prefix not in k
                }
                # When module to save, remove its prefix and discard the original module 当模块需要保存时，删除其前缀并丢弃原始模块
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
            if self.accelerator.is_main_process:
                llm_model = (
                    self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                )
                llm_model.load_weights(state_dict.items())
            # Unmerge the adapter to restore the model to its original state. 解绑适配器以恢复模型的原始状态。
            # This must be done after loading weights to ensure they correspond to the merged state. 必须在加载权重后完成，以确保它们对应于合并的状态。
            if is_peft_model(unwrapped_model): # 检查模型是否是 PEFT 模型
                unwrapped_model.unmerge_adapter()
                # 解绑适配器以恢复模型的原始状态。必须在加载权重后完成，以确保它们对应于合并的状态。
    @profiling_decorator
    def _prepare_inputs( # 用于在训练和评估过程中准备输入数据
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]: 
        mode = "eval" if self.control.should_evaluate else "train"  
        if mode == "train": # 如果模式是训练
            if self.state.global_step % self.num_iterations == 0: # 如果全局步数是迭代次数的倍数
                inputs = self._generate_and_score_completions(inputs)  # 生成并评分完成
                self._buffered_inputs[
                    self._step % self.args.gradient_accumulation_steps
                ] = inputs  # 将输入缓存到缓冲区中
            else: 
                inputs = self._buffered_inputs[ # 从缓冲区中获取输入
                    self._step % self.args.gradient_accumulation_steps
                ]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs. 
            # 在评估中，我们不重用多个更新中的完成，所以我们不需要缓冲输入的数据。
            inputs = self._generate_and_score_completions(inputs) # 生成并评分完成的数据
        return inputs

    # ---------- 生成并评分完成的数据 ----------
    # 这个方法在训练和评估过程中被调用，主要用于生成模型的输出（完成）并计算这些输出的奖励（rewards）
    '''
            "prompt_ids": prompt_ids,   # 模型的输入
            "prompt_mask": prompt_mask,   # 输入的掩码：指示哪些 token 是有效的，在计算损失和评分时，只有有效 token 被考虑。
            "completion_ids": completion_ids,   # 模型输出的预测的 token IDs
            "completion_mask": completion_mask,   # 掩码
            "old_per_token_logps": old_per_token_logps,   # 旧的每个token的对数概率
            "ref_per_token_logps": ref_per_token_logps,   # 这些是对数概率是基于参考模型计算的。参考模型通常是一个预训练模型，用于提供基准性能。
            "advantages": advantages,   # 优势
    '''
    def _generate_and_score_completions( # 生成并评分完成的数据
        self, inputs: dict[str, Union[torch.Tensor, Any]] 
    ) -> dict[str, Union[torch.Tensor, Any]]:


        device = self.accelerator.device
        # 从输入中提取提示（prompts）
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] # 应用chat模板
            for example in inputs
        ]
        prompt_inputs = self.processing_class( # 处理提示文本，生成输入张量
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs) # 准备输入数据

        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation 使用vLLM或常规生成生成完成
        # 如果使用 vLLM，收集所有提示并在主进程中生成完成，然后将完成广播到所有进程
        if self.args.use_vllm:
            # First, have main process load weights if needed 首先，如果需要，主进程加载权重
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm() # 将模型移动到vLLM
                self._last_loaded_step = self.state.global_step # 更新最后加载的步数

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            #  使用vLLM生成完成：收集所有提示并使用它们在主进程中进行单次调用
            all_prompts_text = gather_object(prompts_text) # 收集所有提示，将所有提示文本收集到一个列表中，以便在主进程中进行处理
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate    
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                # 由于 'prompts' 包含 'num_generations' 个重复，我们首先取唯一的提示，并为每个提示生成 num_generations 个输出。这比单独为每个重复的提示生成输出要快。

                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text)) # 取唯一的提示
                all_outputs = self.llm.generate( # 使用vllm生成完成
                    ordered_set_of_prompts,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
                completion_ids = []
                for outputs in all_outputs: # 遍历所有输出
                    for output in outputs.outputs:
                        completion_ids.append(output.token_ids) # 添加token id
            else:
                completion_ids = [None] * len(all_prompts_text) # 如果没有主进程，则设置为None
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0) # 广播完成，将完成从主进程广播到所有进程，确保每个进程收到其对应的切片。
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts), # 切片，确保每个进程收到其对应的切片。
            )
            completion_ids = completion_ids[process_slice] # 从切片中获取完成，确保每个进程收到其对应的切片。

            # Pad the completions, and concatenate them with the prompts 填充完成，并将它们与提示连接起来
            completion_ids = [
                torch.tensor(ids, device=device) for ids in completion_ids
            ] # 将完成转换为张量，并将其移动到设备上
            completion_ids = pad(
                completion_ids, padding_value=self.processing_class.pad_token_id
            ) # 填充完成，并将它们与提示连接起来
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1) # 将提示和完成连接起来
        else:
            # 如果不使用 vLLM，使用常规生成路径生成完成
            with unwrap_model_for_generation( # 解包模型，确保模型在生成过程中使用正确的设备
                self.model, self.accelerator
            ) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate( # 生成完成
                    prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=self.generation_config,
                )

            # Compute prompt length and extract completion ids 计算提示长度，并提取完成
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # 屏蔽所有在第一个EOS token之后的token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full( # 创建一个全0的张量，用于存储EOS token的索引
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)] # 找到EOS token的索引
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand( # 创建一个序列索引，用于存储序列的索引
            is_eos.size(0), -1 # 创建一个序列索引，用于存储序列的索引
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int() # 创建一个完成掩码，用于存储完成掩码

        # 将prompt_mask和completion_mask连接起来，用于计算logits
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(
            1 # 我们只需要计算完成token的logits
        )  # we only need to compute the logits for the completion tokens

        with torch.inference_mode(): # 使用推理模式，确保模型在推理过程中使用正确的设备
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's  
            # 当使用num_iterations == 1时，old_per_token_logps == per_token_logps，所以我们可以跳过它的计算，并使用per_token_logps.detach()代替。
            # computation here, and use per_token_logps.detach() instead.
           
            # 计算每个 token 的对数概率，用于计算KL散度
            if self.num_iterations > 1: #   如果迭代次数大于1
                old_per_token_logps = self._get_per_token_logps( # 获取每个token的对数概率
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep #  使用模型获取每个token的对数概率
                )
            else:
                old_per_token_logps = None # 如果迭代次数等于1，则设置为None


            if self.beta == 0.0:
                ref_per_token_logps = None # 如果beta等于0，则设置为None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps( # 使用模型获取每个token的对数概率
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter(): # 禁用adapter，确保模型在推理过程中使用正确的设备
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids, # 连接提示和完成
                        attention_mask, # 连接提示和完成
                        logits_to_keep, # 连接提示和完成
                    )

        #  解码: 生成的completions
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        # 计算每个奖励函数的输出
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ): # 遍历每个奖励函数和奖励处理类
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models     
                # 如果输入是对话式的，则将提示和完成转换为对话格式
                if is_conversational(inputs[0]):
                    messages = [
                        {"messages": p + c} for p, c in zip(prompts, completions)
                    ]
                    # 将提示和完成转换为文本
                    texts = [
                        apply_chat_template(x, reward_processing_class)["text"]
                        for x in messages
                    ]
                else:
                    # 如果输入不是对话式的，则将提示和完成转换为文本
                    texts = [p + c for p, c in zip(prompts, completions)]
                # 处理奖励输入
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode(): # 使用推理模式，确保模型在推理过程中使用正确的设备
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[
                        :, 0
                    ]  # Shape (B*G,)
                    # 计算每个奖励函数的奖励
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations 
                # 重复所有输入列（但 "prompt" 和 "completion"）以匹配生成数量
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {
                    # 创建奖励函数输入字典: 将提示和完成转换为文本
                    key: [example[key] for example in inputs] for key in keys
                }
                output_reward_func = reward_func(
                    # 创建奖励函数输入字典: 将提示和完成转换为文本
                    prompts=prompts, 
                    completions=completions, 
                    **reward_kwargs
                )
                # 计算每个奖励函数的奖励
                rewards_per_func[:, i] = torch.tensor(
                    # 将奖励转换为张量
                    output_reward_func, dtype=torch.float32, device=device
                )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the completions may be distributed across processes    
        # 收集每个奖励函数的奖励：这一部分至关重要，因为奖励是按组归一化的，并且完成可能分布在不同的进程中
        rewards_per_func = gather(rewards_per_func) 

        # 为每个奖励函数应用权重并求和
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(
            dim=1
        )

        # 计算每个奖励函数的平均值和标准差
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # 归一化奖励，计算优势
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        # 计算优势  (r - mean) / (std + 1e-4)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # 切片，保留本地数据
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # 记录指标
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics[mode]["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item()
            )

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
        ):
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if (
                    self.args.report_to
                    and "wandb" in self.args.report_to
                    and wandb.run is not None
                ):
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,   # 模型的输入
            "prompt_mask": prompt_mask,   # 输入的掩码：指示哪些 token 是有效的，在计算损失和评分时，只有有效 token 被考虑。
            "completion_ids": completion_ids,   # 模型输出的预测的 token IDs
            "completion_mask": completion_mask,   # 掩码
            "old_per_token_logps": old_per_token_logps,   # 旧的每个token的对数概率
            "ref_per_token_logps": ref_per_token_logps,   # 这些是对数概率是基于参考模型计算的。参考模型通常是一个预训练模型，用于提供基准性能。
            "advantages": advantages,   # 优势
        }


    # 核心是:GRPO损失函数的计算
    @profiling_decorator
    def compute_loss( # 计算损失
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model 计算模型每个token的对数概率
        # ---------- 1. 准备输入数据 ----------
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1) 
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1) # 连接prompt_mask和completion_mask
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens 我们只需要计算完成token的logits

        # ---------- 2. 计算模型每个token的对数概率 ----------  
        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )

        # ---------- 3. 计算模型和参考模型之间的KL散度 ----------  
        # Compute the KL divergence between the model and the reference model 计算模型和参考模型之间的KL散度
        if self.beta != 0.0: # 如果beta不等于0，则计算KL散度
            ref_per_token_logps = inputs["ref_per_token_logps"]
            # ---------- 3.1 计算KL散度 ----------  
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # ---------- 4. 计算损失 ----------  
        advantages = inputs["advantages"] # 获取优势    
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead. 当使用num_iterations == 1时，old_per_token_logps == per_token_logps，所以我们可以跳过它的计算，并使用per_token_logps.detach()代替。
        old_per_token_logps = (
            inputs["old_per_token_logps"] #  如果迭代次数等于1，则使用旧的每个token的对数概率
            if self.num_iterations > 1 #    如果迭代次数大于1
            else per_token_logps.detach() #  否则使用每个token的对数概率
        )
        # 原始概率比:
        coef_1 = torch.exp(per_token_logps - old_per_token_logps) 
        # 限制在1 +/- epsilon之间，防止数值不稳定。
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon) # 计算系数2

        per_token_loss1 = coef_1 * advantages.unsqueeze(1) # 计算损失1
        per_token_loss2 = coef_2 * advantages.unsqueeze(1) # 计算损失2
        # 取负值，因为优化器通常最小化损失，而这里我们希望最大化奖励。
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2) 

        if self.beta != 0.0: 
            # 添加KL散度损失，平衡奖励和KL散度。
            # 这里，我们通过将KL散度损失乘以一个权重（beta）来平衡奖励和KL散度。
            # 权重越大，KL散度损失对总损失的影响越大。
            # 这意味着，当KL散度损失较大时，优化器会更多地关注KL散度损失，而不是奖励。
            # 反之，当KL散度损失较小时，优化器会更多地关注奖励。
            # 最终计算的损失
            per_token_loss = per_token_loss + self.beta * per_token_kl 
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # ---------- 5. 记录指标 ----------  
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0: # 如果beta不等于0，则计算KL散度
            mean_kl = (
                # completion_mask ：一个掩码张量，用于指示哪些 token 是有效的。它的形状与 per_token_kl 相同，
                # 确保只计算有效 token 的 KL 散度。
                (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
            ).mean()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )
        return loss

    # prediction_step -> _prepare_inputs -> compute_loss -> log
    def prediction_step( # 预测步骤
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys: Optional[list[str]] = None,
    ):
        # 在预测和评估阶段，我们需要准备输入数据，以便在计算损失时使用。
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
                # detach()：从计算图中分离损失张量，使其不再需要梯度。这在预测和评估阶段是必要的，因为不需要对损失进行反向传播。
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None: # 日志
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

    def create_model_card( # 创建模型卡
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

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(
            self.model.config._name_or_path
        ):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=(
                wandb.run.get_url()
                if is_wandb_available() and wandb.run is not None
                else None
            ),
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
