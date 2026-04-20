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
                    new_batch = new_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
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

                        new_batch.batch["token_level_scores"] = reward_tensor

                        # check 2k filtered acc
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
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"]
                                .sum(dim=-1)
                                .numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"]
                                .sum(dim=-1)
                                .numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"],
                            new_batch.non_tensor_batch[metric_name],
                            strict=True,
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(
                            new_batch.non_tensor_batch["uid"]
                        ):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = (
                            new_batch
                            if batch is None
                            else DataProto.concat([batch, new_batch])
                        )

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:  # 样本收集不足
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = (
                                self.config.algorithm.filter_groups.max_num_gen_batches
                            )
                            if (
                                max_num_gen_batches <= 0
                                or num_gen_batches < max_num_gen_batches
                            ):
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # 样本收集足够：Align the batch
                            traj_bsz = (
                                self.config.data.train_batch_size
                                * self.config.actor_rollout_ref.rollout.n
                            )
                            batch = batch[:traj_bsz]

                    # === Updating 更新阶段 ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        # 计算就概率
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
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
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item()
                        }
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:  # 使用参考策略
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                batch
                            )
                            batch = batch.union(ref_log_prob)

                    # compute values 计算价值
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )
                        # 计算优势
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
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
                    global_log_buffer = None
                    # 默认值为0
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)

                        # Handle global logging for negative_approx_kl
                        if "global_log_buffer" in actor_output.non_tensor_batch:
                            global_log_buffer = actor_output.non_tensor_batch.pop(
                                "global_log_buffer"
                            )

                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # NOTE: We copy the rollout saving from ray_ppo_trainer.py#L1282.
                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer(
                            "dump_rollout_generations", timing_raw, color="green"
                        ):
                            inputs = self.tokenizer.batch_decode(
                                batch.batch["prompts"], skip_special_tokens=True
                            )
                            outputs = self.tokenizer.batch_decode(
                                batch.batch["responses"], skip_special_tokens=True
                            )
                            scores = (
                                batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            )
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get(
                                    "ground_truth", None
                                )
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                        # Save negative_approx_kl to the same directory
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
                                    neg_kl = item["negative_approx_kl"]
                                    resps = item["responses"]

                                    if isinstance(neg_kl, torch.Tensor):
                                        neg_kl = neg_kl.tolist()
                                    if isinstance(resps, torch.Tensor):
                                        resps = resps.tolist()

                                    # Try to flatten if possible, or just save as batch
                                    # If neg_kl is a list of values and resps is list of responses
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
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
