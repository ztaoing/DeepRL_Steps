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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # 恢复训练状态
        # load checkpoint before doing anything 加载模型检查点
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            """
            断言检查：代码中显式地检查了 val_metrics 是否存在。这说明开发者认为“训练前的验证结果必须有效”是继续训练的前提条件。
            """
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")

            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        """
        这是一个“跳过采样（Rollout）”的开关机制。当配置开启时，模型不会在当前步骤重新生成新的回答（Response），
        而是复用之前的逻辑或采用某种优化策略（如投机解码）
        """
        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            # 包装/劫持生成函数。
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,  # 进度条总长度
            initial=self.global_steps,
            desc="Training Progress",
        )

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        # “按需分析，减少开销”
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.trainer.profile_steps
            if self.config.trainer.profile_steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                # 使用 with 语句可以确保无论代码是否发生异常，计时结束和资源清理（如 NVTX 范围的闭合）都能自动执行，保证代码的健壮性和可读性
                with marked_timer("start_profile", timing_raw):
                    """
                    效果：这是一个快照模式。比如你设定在第 100 步分析，那么只有在第 100 步时，_start_profiling(True) 会被调用，其他时候都是 False

                    1. curr_step_profile 为 True：说明当前步（比如第 101 步）在分析计划内。
                    2. not prev_step_profile 为 True：说明上一步（比如第 100 步）不在分析计划内（或者被强制视为 False）。
                    3. 组合含义：只有当“上一步没分析”且“当前步要分析”时，才调用 _start_profiling

                    避免重复启动：性能分析器（Profiler）通常只需要启动一次。如果你要连续分析 5 步，你不希望在第 100、101、102... 步每次都调用 start。你只需要在序列的第一步（转折点）启动它。
                    确保覆盖：这个逻辑确保了在连续分析周期的起始点，分析器被正确激活。
                    """
                    self._start_profiling(
                        not prev_step_profile
                        and curr_step_profile  # 说明上一步（比如第 100 步）不在分析计划内（或者被强制视为 False）。
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                # 将原始的数据字典转换为框架内部通用的标准数据协议对象。
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                """
                根据数据的模态类型（纯文本 vs 多模态），将 DataProto 中的数据进行拆分，提取出专门用于生成阶段（Generation/Rollout）的数据包
                1.在纯文本训练中，数据通常只包含 input_ids（文本的数字化表示）。
                2.在多模态训练（如 LLaVA、Qwen-VL）中，数据除了文本，还包含图片张量、图片占位符等信息，这些通常被存放在 multi_modal_data 字段中。
                """
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                """
                gen_batch 是剥离出来的“纯净”输入数据（包含 input_ids, attention_mask 等

                为了让模型对同一个 Prompt 生成 n 个不同的回答（用于 PPO/GRPO 的多样性采样），我们需要在输入端把这个 Prompt 复制 n 份
                """
                gen_batch = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(
                            gen_batch
                        )
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = (
                                self.actor_rollout_wg.generate_sequences(
                                    gen_baseline_batch
                                )
                            )

                            new_batch = new_batch.union(gen_baseline_output)
                            #
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(
                                batch_keys=list(gen_baseline_output.batch.keys())
                            )

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
                        dtype=object,
                    )

                    """
                    对象：new_batch 是主数据容器，此时它主要包含元数据（如 uid、原始文本等），因为核心的 Tensor 数据之前已经被 pop 走了。
                    目的：数据对齐（Alignment）。
                        - 此时，gen_batch_output（模型生成的回答）已经有了 Batch_Size * n 条数据。
                        - 但是，new_batch 里的元数据（比如用于标识样本的 uid）还只有 Batch_Size 条。
                        - 如果直接合并（union），会因为维度不匹配（一个长一个短）而报错。
                        - 因此，必须把 new_batch 里的元数据也复制 n 份，确保每一条生成的回答都能对应到正确的 uid 和原始 Prompt。
                    结果：
                        - new_batch 的长度从 Batch_Size 膨胀到 Batch_Size * n。
                        - 随后通过 new_batch.union(gen_batch_output) 将生成的回答“贴”回主数据包中。
                    """
                    # repeat to align with repeated responses in rollout
                    # 此时的new_batch已经包含了uid，所以这里需要重复n次
                    new_batch = new_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )

                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if (
                            self.use_rm
                        ):  # 如果使用 reward model，则先计算 reward model 的得分
                            # we first compute reward model score 给模型生成的回复打分
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(
                                reward_tensor
                            )  # 将得分结果合并到 new_batch 中

                        # we combine with rule-based rm 规则奖励
                        """
                        格式检查（是否包含 <answer> 标签）。
                        代码正确性（单元测试是否通过）。
                        长度惩罚（回答是否太长）。
                        """
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get(
                                "reward_extra_info", {}
                            )
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = (
                            reward_tensor  # 将得分结果合并到 new_batch 中
                        )

                        # check 2k filtered acc
                        """
                        为了支持 DAPO 算法中的动态采样过滤机制（Dynamic Sample Filtering）。
                        为了维持 Batch 多样性，会过滤掉那些所有样本奖励都一样的 Prompt。

                        这段代码的主要作用是将奖励计算过程中产生的“额外元数据”（如准确率、格式正确性、规则匹配结果等）保存回数据包中。

                        在强化学习中，除了计算用于更新梯度的 reward_tensor（奖励张量），我们通常还需要记录一些辅助信息，例如：
                            1. Acc (准确率)：回答是否正确（例如数学题答案是否匹配）。
                            2. Format Check：回答是否符合特定格式（如是否包含 </think>）。
                            3. Filter Mask：该样本是否应该被保留。
                        这些信息不需要参与反向传播（不需要在 GPU 上作为 Tensor 计算），但需要在 CPU 端用于日志记录或筛选数据。
                        因此，它们被存放在 reward_extra_infos_dict 中，并最终更新到 new_batch.non_tensor_batch。

                        如果某个 Prompt 对应的 4 个样本 acc 都是 0（或都是 1），标准差为 0，
                        说明模型在这个问题上没有“探索”出多样性，这个 Prompt 就会被过滤（Filter）掉，不参与梯度更新。

                        
                        它把奖励函数计算出来的辅助判据（如准确率、规则匹配情况），
                        从临时的字典 reward_extra_infos_dict 搬运到正式的数据包 new_batch 中，
                        以便后续的 DAPO 过滤算法 能够读取这些指标，决定保留哪些样本进行训练。

                        """
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {
                                    k: np.array(v)
                                    for k, v in reward_extra_infos_dict.items()
                                }
                            )

                        # KL 惩罚 compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch[
                                "token_level_scores"
                            ]
                    """
                    过滤机制：
                    1. 根据标准差或分数过滤掉不好的样本
                    2. 如果过滤后样本不足，会继续生成直到凑够 batch size
                    
                    """
                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:
                        # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        # 过滤后如果样本不足，会继续生成直到凑够 batch size
                        """
                        将模型输出的 Token 级数据（Tensor）转换为序列级指标（Numpy），以便后续根据这些指标对样本进行分组和过滤。

                        在强化学习（如 PPO/GRPO）中，模型输出的是 Token 级别 的数据（例如每个字对应的奖励或分数）。
                        但在 DAPO 算法中，我们需要以 序列（Sequence/Prompt） 为单位来评估样本的质量。
                        - 输入：new_batch.batch 中存的是 Tensor，形状通常是 [Batch_Size, Sequence_Length]。
                        - 目标：我们需要计算整个句子的总分，并将其移动到 CPU 内存（non_tensor_batch）中，因为后续的过滤逻辑（如计算标准差、筛选）通常在 CPU 上进行，且不需要梯度。
                        """
                        metric_name = (
                            self.config.algorithm.filter_groups.metric
                        )  # 决定使用哪种类型的指标分数来衡量样本的好坏。DAPO 允许用户灵活配置，比如是用“最终奖励”还是“原始分数”。
                        if (
                            metric_name == "seq_final_reward"
                        ):  # 如果使用的是最终奖励，则计算每个样本的最终奖励
                            # Turn to numpy for easier filtering
                            """
                            数据源：token_level_rewards。这通常包含了 KL 散度惩罚 后的奖励（即 R_final=R_model−β⋅KL ）。
                            场景：如果你希望过滤机制考虑到 KL 散度（即避免模型为了刷分而偏离原始分布太远），你会选择这个指标。
                            """
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"]
                                .sum(dim=-1)
                                .numpy()
                            )
                        elif (
                            metric_name == "seq_reward"
                        ):  # 如果使用的是原始序列奖励，则计算每个样本的序列奖励
                            """
                            数据源：token_level_scores。这通常是 原始奖励模型（Reward Model）的输出 或 规则奖励，尚未减去 KL 散度。
                            场景：如果你只关心模型是否完成了任务（如代码通过测试、答案正确），而不关心 KL 惩罚，可以选择这个指标。
                            """
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"]
                                .sum(dim=-1)
                                .numpy()
                            )

                        """

                        动态采样: 它的核心目的是：剔除那些“没有区分度”的 Prompt，只保留那些能让模型学到东西的样本进行训练。

                        如果一个 Prompt 生成的多个回答得分都一样（比如全对或全错），模型无法从中通过对比来学习，因此这段代码会将这些样本过滤掉。

                        """
                        # Collect the sequence reward for each trajectory
                        """
                        按 Prompt ID 收集分数，以便后续计算标准差。

                        此时的 new_batch 是扁平的，里面混合了所有 Prompt 的生成结果（例如：A1, B1, A2, B2...）

                        通过 zip 遍历每个样本的 uid（Prompt 的唯一标识）和对应的 metric_val（之前计算好的序列总分）
                        结果：将数据重组为 { "Prompt_A": [0.5, 0.8, 0.2], "Prompt_B": [1.0, 1.0, 1.0] } 的形式。即：每个 Prompt 对应它生成的 N 个样本的分数列表。
                        """
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"],
                            new_batch.non_tensor_batch[metric_name],
                            strict=True,
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        """
                        计算标准差：衡量“区分度”

                        计算每个 Prompt 对应分数列表的标准差（Standard Deviation, std）

                        std > 0：说明这组样本分数有高有低。模型对这个问题产生了多样化的回答，且奖励模型能区分好坏。这是有效的训练信号。
                        std == 0：说明这组样本分数完全一样（例如全是 0 或全是 1）。模型在这个问题上要么完全不会（全错），要么已经饱和（全对），或者奖励模型失效。这种情况下，梯度更新没有意义（或者是零梯度）。
                        """
                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        """
                        过滤样本：只保留那些“有区分度”的 Prompt
                        过滤规则：保留满足以下任一条件的 uid：
                         1. std > 0：分数有波动，值得学习。
                         2. len(...) == 1：该 Prompt 只有一个样本。这种情况无法计算组内对比，通常默认保留（或者在 GRPO 中作为特殊情况处理）
                        
                        """
                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        """
                        索引映射：从 Prompt 回到样本

                        前面我们筛选出的是“合格的 Prompt ID”。
                        但我们需要操作的是 new_batch 中的“样本行”。
                        这段代码遍历原始 Batch，找出所有属于“合格 Prompt”的样本的索引位置（index）
                        """
                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(
                            new_batch.non_tensor_batch["uid"]
                        ):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        """
                        切片与累积：构建最终训练包

                        切片：new_batch[kept_traj_idxs] 利用上一步的索引，直接剔除掉那些“无区分度”的样本。
                        累积：
                         - DAPO 的过滤可能会导致单个生成批次的数据量不足。
                         - 因此，代码将过滤后的数据累加到 batch 变量中。
                         - 后续逻辑（未展示）通常会检查 batch 的大小是否达到配置的 train_batch_size。如果不够，会继续生成下一批数据；如果够了，就进行梯度更新。
                        """
                        new_batch = new_batch[kept_traj_idxs]
                        batch = (
                            new_batch
                            if batch is None
                            else DataProto.concat([batch, new_batch])
                        )

                        """
                        通常会检查 batch 的大小是否达到配置的 train_batch_size。如果不够，会继续生成下一批数据；
                        如果够了，就进行梯度更新。

                        """
                        prompt_bsz = self.config.data.train_batch_size

                        # num_prompt_in_batch 是当前经过过滤后实际保留的 Prompt 数量。
                        # 如果保留下来的有效 Prompt 数量少于目标值，说明还需要继续生成数据。
                        if num_prompt_in_batch < prompt_bsz:  # 样本收集不足
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = (
                                self.config.algorithm.filter_groups.max_num_gen_batches
                            )
                            """

                            max_num_gen_batches <= 0：如果设置为 0 或负数，表示“无限重试”（Endless trials）。无论数据多难，都要一直生成直到凑够 Batch。
                            num_gen_batches < max_num_gen_batches：如果设置了正数（例如 10），且当前尝试次数还没达到上限，则允许继续。
                            
                            """
                            if (
                                max_num_gen_batches <= 0
                                or num_gen_batches < max_num_gen_batches
                            ):  # 如果 max_num_gen_batches <= 0，则允许无限生成；否则，检查是否达到最大生成次数
                                print(f"{num_gen_batches=}. Keep generating...")
                                # 打印日志，更新进度条，增加生成步数计数，然后执行 continue。
                                progress_bar.update(1)
                                self.gen_steps += 1
                                continue
                            else:
                                """

                                触发条件：如果已经尝试了 max_num_gen_batches 次（例如 10 次），但收集到的有效样本依然不足。
                                型怎么跑都是全错，或者奖励模型怎么都给一样的分，导致样本全被过滤掉了）。
                                处理：主动报错并停止训练，提示用户检查数据或开启无限重试模式

                                """
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # 样本收集足够：Align the batch （进入“训练”模式）
                            traj_bsz = (
                                self.config.data.train_batch_size
                                * self.config.actor_rollout_ref.rollout.n
                            )
                            batch = batch[:traj_bsz]

                    # === Updating 更新阶段 ===
                    """
                    
                    从“数据准备阶段”正式进入了“模型更新阶段”。

                    它的核心任务是：为计算梯度和损失函数做最后的张量准备。具体来说，它需要计算响应掩码、处理分布式训练中的数据平衡，
                     并重新计算旧策略的对数概率（old_log_prob），这是 PPO/DAPO 算法计算重要性采样比率（Importance Sampling Ratio）的基础。

                    """

                    """

                    目的：明确哪些 Token 是模型生成的（需要计算 Loss），哪些是 Prompt 自带的（不需要计算 Loss）。
                    逻辑：通常根据 attention_mask 和 input_ids 的位置信息生成。只有回答部分的 Token 掩码为 1，其余为 0。

                    """
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.

                    """
                    分布式数据平衡：确保每个进程中的有效 Token 数量大致相同。
                    
                    背景：在分布式数据并行（DDP）训练中，不同 GPU 上的 Batch 可能包含长度差异巨大的序列。这会导致“长尾效应”——计算速度取决于最慢的那个 GPU（即序列最长的那个）。
                    动作：_balance_batch 会打乱并重新分配不同 GPU 之间的样本，使得每个 GPU 处理的有效 Token 总数大致相等。

                    副作用（代码注释重点）：
                     - 这会改变数据的顺序。
                     - 不影响优势计算：因为优势函数（Advantage）是基于 uid（样本ID）计算的，顺序乱了没关系，只要 ID 对得上就行。
                     - 可能影响 Loss：因为后续的 Mini-batch 切分是基于当前顺序的。如果顺序变了，组成 Mini-batch 的样本组合也会变，可能导致 Loss 的微小波动。

                    """
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    """
                    统计全局 Token 数量

                    目的：记录当前 Batch 中实际包含多少个有效 Token。
                    用途：通常用于日志记录（Logging），或者在计算平均 Loss 时作为分母，确保梯度的缩放是正确的。
                    
                    """
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # recompute old_log_probs 计算旧策略的对数概率（old_log_prob）
                    with marked_timer("old_log_prob", timing_raw, "blue"):

                        """

                        - 含义：我们需要知道当前模型（在更新前）生成这些回答的概率是多少。
                        - 为什么叫“旧”：因为接下来模型参数会更新，更新后的模型生成的概率叫“新概率”。PPO 算法的核心就是比较“新概率”和“旧概率”的比值，来限制更新幅度（Clip操作）。

                        """
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

                        """

                        熵：衡量模型输出的不确定性。熵越高，模型越困惑；熵越低，模型越确定。
                        agg_loss：根据配置（如 token_mean 或 sequence_mean）将每个 Token 的熵聚合成一个标量值。

                        """
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = (
                            self.config.actor_rollout_ref.actor.loss_agg_mode
                        )
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=loss_agg_mode,
                        )

                        """
                        记录：将熵值存入 metrics 字典，用于后续打印日志（监控模型是否过早收敛）。
                        清理：pop("entropys") 移除临时计算的熵数据，节省显存。
                        合并：batch.union(old_log_prob) 将计算好的 old_log_prob 张量合并回主数据包。
                             此时，batch 中包含了生成时的回答、奖励、优势函数以及现在的旧策略概率，集齐了计算 Loss 所需的所有要素。
                        """
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item()
                        }
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    """
                    什么是 Reference Policy？
                        - 它通常是强化学习训练开始前的初始模型（即 SFT 模型）的一个冻结副本。
                        - 在整个训练过程中，它的参数不会更新。

                    为什么要计算它的 Log Prob？
                        - 核心目的：防止模型崩溃（Mode Collapse）和奖励刷分（Reward Hacking）。
                        - 在计算 Loss 时，我们会计算当前 Actor 模型输出与 Reference 模型输出的 KL 散度。
                        - 如果 Actor 偏离初始模型太远（比如开始说胡话但骗过了奖励模型），KL 惩罚项会变大，强行把模型拉回来。
                    """
                    if self.use_reference_policy:  # 使用参考策略
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                batch
                            )  # 调用 ref_policy_wg（参考策略工作流组）对当前的 Batch 数据进行前向传播。
                            batch = batch.union(ref_log_prob)

                    # compute values 计算价值
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    """
                    batch 数据包中已经集齐了计算 PPO Loss 所需的所有核心要素:
                                       来源	                      用途
                    Response	    Rollout 生成	           训练的目标文本
                    Reward	        Reward Model / Rules	  告诉模型“做得好不好”
                    Old Log Prob	Actor (上一轮)	           计算重要性采样比率 (Ratio)
                    Ref Log Prob	Reference Model (本段代码)	计算 KL 散度，防止模型跑偏
                    Values	        Critic (本段代码)	        计算优势函数 (Advantage)，作为基准线
                    """

                    """
                    “优势函数（Advantage）计算”的入口 :它决定了梯度更新的方向




                    """
                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        """
                        含义：这是一个专门针对 GRPO 算法的配置。
                        作用：决定是否在计算优势时，除以组内奖励的标准差（Standard Deviation）。这相当于一种归一化操作，防止奖励数值过大导致训练不稳定。
                        """
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )
                        # 计算优势
                        """
                        输入：包含 Rewards、Values、Old Log Probs 的 batch。
                        输出：增加了 advantages（优势）和 returns（回报）张量的 batch。
                        """
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,  # 根据传入的 adv_estimator 参数，compute_advantage 函数内部会执行不同的逻辑：
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # 更新 update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    """
                    
                    什么是 Critic Warmup？
                     在 PPO 训练的初期，Actor（策略）和 Critic（价值网络）都在快速变化。
                     - 问题：如果 Critic 还没训练好，它预测的 Values 就不准确。此时如果 Actor 基于这些错误的 Values 计算出的 Advantage（优势）去更新自己，Actor 就会被“误导”，导致训练崩塌。
                     - 策略：设置一个 critic_warmup 步数（例如 100 步）。在这之前，只更新 Critic，不更新 Actor。让 Critic 先学会“怎么打分”，等它稍微靠谱一点了，再让 Actor 开始学习“怎么行动”。
                    
                     代码逻辑
                     - 检查当前步数 self.global_steps 是否超过了预热阈值 self.config.trainer.critic_warmup。
                     - 如果没超过：跳过 Actor 更新，只更新 Critic（Critic 更新代码通常在后面，未在此片段展示）。
                     - 如果超过了：进入 if 块，开始更新 Actor。
                    """
                    global_log_buffer = None
                    # 默认值为0
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # 更新 actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            """
                            update_actor(batch)：这是整个训练循环的高潮。
                            输入：包含了 Prompt、Response、Reward、Old Log Prob、Advantage 等所有信息的 batch。

                            内部逻辑（未展示，但在函数内）：
                                1.计算 PPO Loss（包含 Policy Loss 和 KL Penalty）。
                                2.执行反向传播（Backpropagation）。
                                3.更新 Actor 的模型参数（Weights）

                            输出：actor_output，通常包含更新后的指标（Metrics）和一些需要全局同步的日志数据。
                            """
                            actor_output = self.actor_rollout_wg.update_actor(batch)

                        # Handle global logging for negative_approx_kl
                        """
                        在分布式训练（DDP）中，每个 GPU 计算出的指标（如 Loss）通常会被平均。
                        但是，有些指标（特别是 KL 散度）可能需要更精细的处理，
                        或者需要在所有 GPU 之间进行特殊的聚合（例如计算全局的 KL 分布，而不仅仅是平均值）。
                        
                        代码逻辑
                            - 检查 actor_output 中是否包含 global_log_buffer。
                            - 如果包含，将其提取出来并存入临时变量 global_log_buffer，同时从 non_tensor_batch 中移除（pop），防止它被当作普通指标重复处理。
                            - 这个 buffer 稍后可能会被用于更高级的日志记录或监控。
                        """
                        if "global_log_buffer" in actor_output.non_tensor_batch:
                            global_log_buffer = actor_output.non_tensor_batch.pop(
                                "global_log_buffer"
                            )

                        """
                        
                        步骤
                        1. reduce_metrics：在分布式环境下，将所有 GPU 上的指标进行平均（All-Reduce 操作），得到全局指标。
                        2. metrics.update：将处理好的指标存入总的 metrics 字典，以便后续打印到控制台或写入 TensorBoard/WandB。
                        
                        """
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    """
                    这段代码主要负责训练过程的“可视化”与“数据留存”

                    它的作用是将当前批次中模型的输入（Prompt）、输出（Response）、得分（Score）以及 KL 散度等关键信息保存到磁盘。
                    这对于调试模型行为、分析 Bad Case 以及复现训练过程至关重要。

                    """
                    # NOTE: We copy the rollout saving from ray_ppo_trainer.py#L1282.
                    # Log rollout generations if enabled
                    """
                    首先检查配置中是否设置了 rollout_data_dir。
                     目的：这是一个可选功能。如果用户没有指定保存路径，则跳过所有 IO 操作，节省训练时间。
                    """
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        # 为了让人类可读，需要将 Tensor 格式的数据转换回文本。
                        with marked_timer(
                            "dump_rollout_generations", timing_raw, color="green"
                        ):
                            # 1. 解码文本
                            inputs = self.tokenizer.batch_decode(
                                batch.batch["prompts"], skip_special_tokens=True
                            )
                            outputs = self.tokenizer.batch_decode(
                                batch.batch["responses"], skip_special_tokens=True
                            )

                            # 2. 计算总分：将每个dede的的分数求和，得到整个句子的总分，并转python列表
                            scores = (
                                batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            )
                            # 3. 提取标准答案 ground_truth
                            #  尝试从非张量数据中获取该问题的标准答案，用于后续对比模型回答是否正确
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get(
                                    "ground_truth", None
                                )
                                for item in batch
                            ]
                            """
                            处理额外元数据 (Request ID)，如果存在的话
                             目的：保留 request_id。这在分布式训练或异步生成中非常重要，用于追踪某个特定的生成任务属于哪个请求，方便日志关联。
                              """
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )
                            """
                            保存生成内容 (Dump Generations):
                             动作：调用内部方法 _dump_generations。
                             结果：通常会将 Prompt、模型回答、得分、标准答案 写入到一个 JSONL 文件中。
                             用途：你可以打开这个文件，直观地看到模型在训练过程中到底说了什么，为什么得分高或低
                            """
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                        # Save negative_approx_kl to the same directory
                        """
                        这部分代码专门用于保存 KL 散度（负对数似然比），这是衡量模型是否偏离初始策略的重要指标。
                        """
                        if global_log_buffer is not None:
                            import json
                            import os

                            save_path = os.path.join(
                                rollout_data_dir,
                                f"negative_approx_kl_global_step_{self.global_steps}.jsonl",
                            )

                            with open(save_path, "w") as f:
                                for item in global_log_buffer:
                                    # item is { "negative_approx_kl": tensor, "responses": tensor }
                                    # 提取张量数据
                                    neg_kl = item["negative_approx_kl"]
                                    resps = item["responses"]

                                    # 类型转换：Tensor -> List
                                    if isinstance(neg_kl, torch.Tensor):
                                        neg_kl = neg_kl.tolist()
                                    if isinstance(resps, torch.Tensor):
                                        resps = resps.tolist()

                                    # Try to flatten if possible, or just save as batch
                                    # If neg_kl is a list of values and resps is list of responses
                                    """
                                    序列化逻辑：
                                    如果是扁平列表（即每个样本对应一个值），则逐行写入 JSONL

                                    扁平化处理：代码尝试判断数据是否是“样本级”的（即 len(neg_kl) == len(resps)）。
                                              如果是，它会将其展开为标准的 JSONL 格式（每行一个样本），方便使用 jq 或 pandas 进行后续分析
                                    """
                                    if (
                                        isinstance(neg_kl, list)
                                        and isinstance(resps, list)
                                        and len(neg_kl) == len(resps)
                                        and not isinstance(neg_kl[0], list)
                                    ):
                                        for nk, r in zip(neg_kl, resps):
                                            json.dump(
                                                {
                                                    "negative_approx_kl": nk,
                                                    "response": r,
                                                },
                                                f,
                                            )
                                            f.write("\n")
                                    else:
                                        json.dump(
                                            {
                                                "negative_approx_kl": neg_kl,
                                                "responses": resps,
                                            },
                                            f,
                                        )
                                        f.write("\n")

                    # validate
                    """
                    负责执行模型评估和状态保存。这是深度学习训练流程中确保模型效果可追踪、训练可恢复的两个关键环节。

                    触发条件:
                     评估并非每一步都进行，因为它耗时较长。代码通过一个组合条件来判断是否触发：
                        1.评估函数存在：self.val_reward_fn is not None。必须定义了用于评估的奖励函数。
                        2.评估频率已启用：self.config.trainer.test_freq > 0。配置中的评估频率必须是一个正数。
                        3.到达评估时机：is_last_step or self.global_steps % self.config.trainer.test_freq == 0。 
                                      当训练到达最后一步，或者当前步数是评估频率的整数倍时（例如每训练1000步），触发评估。
                    """
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):  # 在训练过程中定期或在训练结束时，对模型进行验证，以评估其当前的性能。

                        """
                        执行与记录:
                            执行评估：在 marked_timer 的计时上下文中，调用 self._validate() 方法。这个方法会遍历验证集，计算模型的性能指标（如准确率、奖励分数等）。
                            保存最终指标：if is_last_step: last_val_metrics = val_metrics。如果这是最后一步，会将评估结果单独保存一份，方便训练结束后直接获取模型的最佳性能。
                            合并日志：metrics.update(val_metrics)。将本次评估得到的 val_metrics（如 val/accuracy）合并到主日志字典 metrics 中，以便后续统一打印或记录到 TensorBoard。

                        """
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = (
                                self._validate()
                            )  # 执行模型验证，并返回验证结果
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    """
                    这段逻辑负责将模型的当前状态保存到磁盘，形成一个“检查点”（Checkpoint）。

                    触发条件:
                        与评估类似，保存检查点也有其触发条件：
                        1.保存频率已启用：self.config.trainer.save_freq > 0。
                        2.到达保存时机：is_last_step or self.global_steps % self.config.trainer.save_freq == 0。
                                     在训练的最后一步，或每隔指定的步数进行保存。

                    执行保存:
                        调用保存方法：在计时上下文中，调用 self._save_checkpoint()。
                        保存内容：虽然具体实现在 _save_checkpoint 方法内部，但一个标准的检查点通常包含：
                            模型参数 (model.state_dict())：模型学习到的权重。
                            优化器状态 (optimizer.state_dict())：优化器的动量、学习率等状态，用于精确恢复训练。
                            当前步数 (global_steps)：记录训练进度，以便下次从正确的地方继续。
                            其他元数据：可能还包括最佳性能指标等。
                    核心目的:
                        断点续训：如果训练因意外中断，可以从最近保存的检查点恢复，无需从头开始。
                        版本管理：定期保存的检查点相当于模型的“存档”，可以用于回溯和比较不同训练阶段的模型效果。
                        模型部署：训练完成后，最终保存的检查点就是可以用于推理的最终模型。

                    """
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()
                """
                这段代码是训练循环中性能分析器（Profiler）的生命周期管理部分,主要负责优雅地停止当前步骤的性能分析，并为下一步的分析做好准备。

                核心目的是实现一种按需、按需触发的性能分析机制，而不是在整个训练过程中持续不断地收集数据，从而避免产生巨大的性能开销和日志文件

                在长时间训练中，持续开启性能分析器（Profiler）会带来显著的性能损耗，并生成海量的数据。因此，通常只需要在特定的关键步骤（如训练开始、中间、结束时）进行短暂的“快照”分析。



                """
                with marked_timer("stop_profile", timing_raw):
                    """
                    - 检查配置项 profile_steps 是否被设置。这通常是一个包含需要分析的步数的列表，例如 [0, 100, 500]。
                    - 如果已设置，则检查global_steps + 1 是否在这个列表中。
                    - 最终，next_step_profile 是一个布尔值，True 表示下一步也要分析，False 表示下一步不分析。
                    """
                    next_step_profile = (
                        self.global_steps + 1
                        in self.config.trainer.profile_steps  # 提前判断下一个训练步（global_steps + 1）是否也需要进行性能分析
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    """
                    这段代码实现了一个灵活且高效的性能分析调度器：

                    决定何时停止分析:
                    这是整个逻辑的核心，它根据一个配置项 profile_continuous_steps 来决定停止分析的策略。
                           
                    场景A：非连续分析模式 (profile_continuous_steps 为 False)
                        停止条件：curr_step_profile
                        行为：只要当前步骤被标记为需要分析，就在本步结束时立即停止分析器。这是一种“单步快照”模式。例如，你配置了分析第100步，那么分析器会在第100步开始时启动，在第100步结束时停止，生成一个独立的性能报告。
                    场景B：连续分析模式 (profile_continuous_steps 为 True)
                        停止条件：curr_step_profile and not next_step_profile
                        行为：只有当当前步骤需要分析，但下一步不需要时，才停止分析器。
                        目的：这种模式用于分析一个连续的步骤区间。例如，如果你想分析从第100步到第105步这6个步骤的整体性能，你可以将 profile_steps 设置为 range(100, 106)。
                            - 在第100步，curr_step_profile 为 True，next_step_profile 也为 True（因为101也在列表中），所以不停止。
                            - ...中间步骤同理...
                            - 在第105步，curr_step_profile 为 True，但 next_step_profile 为 False（因为106不在列表中），此时条件满足，分析器停止。
                        这样，你就得到了一个覆盖了第100到105步的、连续的性能分析报告。

                    """
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                    """
                    更新状态，为下一步循环做准备:

                    目的：更新状态变量，以便在下一次训练循环迭代中使用。
                        - curr_step_profile 被更新为刚刚计算出的 next_step_profile。这样，在下一个循环开始时，它就知道自己是否需要启动分析器。
                        - prev_step_profile 记录了上一步的状态，虽然在此代码片段中未直接使用，但可能在其他地方（如日志记录）用于判断状态变化。
                    """
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                """
                这段代码位于训练主循环的末尾，是整个训练流程的“收尾与重置”阶段。
                它负责收集本步的所有指标、记录日志、清理临时变量，并为下一次迭代做准备。


                
                """
                # collect metrics
                """
                收集与训练数据本身相关的指标。
                 内容：可能包括奖励的平均值/标准差、响应序列的平均长度、KL散度等。use_critic 参数用于决定是否需要计算与 Critic 模型相关的指标
                """
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                """
                分析训练流程中各个阶段的耗时。
                 内容：利用之前通过 marked_timer 记录的 timing_raw 字典，计算出数据生成、模型更新、通信等每个环节花费的时间，帮助定位性能瓶颈。
                """
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )

                """
                衡量训练效率。
                 内容：计算每秒处理的样本数（samples/sec）、每秒处理的Token数（tokens/sec）等。n_gpus 用于计算多卡环境下的整体吞吐量。
                """
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                """
                
                在记录完日志后，需要清理当前步骤的临时状态，为下一次循环做准备。
                - timing_raw: 清空耗时记录字典，以便下一轮循环重新计时。
                - batch: 将数据批次置为 None，释放显存。
                - num_prompt_in_batch / num_gen_batches: 重置计数器。
                
                这些变量在 DAPO 的动态采样逻辑中用于累积有效样本，每一步开始时都需要归零。
                """
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                """
                将汇总好的 metrics 字典发送给配置的日志后端（如 TensorBoard, WandB）。
                 效果：你可以在可视化工具中看到实时的 Loss 曲线、奖励变化、吞吐量等图表。

                """
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                """
                逻辑：如果当前是最后一个训练步，则打印最终的验证指标，关闭进度条，并直接 return 退出整个训练函数。
                """
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                """
                逻辑：如果不是最后一步，则更新进度条，并将全局训练步数 global_steps 和生成步数 gen_steps 加一，然后进入下一次 while 循环。
                """
                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
