# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | AlgoConfig],  # config
    ],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}

"""
1. 最外层函数 register_policy_loss(name)：
    职责：接收配置参数（这里是损失函数的名字 name）。
    返回：返回中间的装饰器函数。
    场景：当你写 @register_policy_loss("ppo") 时，调用的就是这一层。
2. 中间层函数 decorator(func)：
    职责：接收被装饰的函数（即你写的具体损失函数实现 func）。
    动作：执行注册逻辑 —— POLICY_LOSS_REGISTRY[name] = func。它把函数的名字和函数本身存入一个全局字典 POLICY_LOSS_REGISTRY 中。
    返回：返回原始函数 func（通常装饰器不改变原函数的功能，只是给它“贴个标签”）。
3. 最内层（实际使用）：
    这是你实际编写的损失函数代码。
"""


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estima

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert (
            kl_ctrl.horizon > 0
        ), f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(
            init_kl_coef=kl_ctrl.kl_coef,
            target_kl=kl_ctrl.target_kl,
            horizon=kl_ctrl.horizon,
        )
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = (
                values[:, t] * response_mask[:, t]
                + (1 - response_mask[:, t]) * nextvalues
            )
            lastgaelam = (
                lastgaelam_ * response_mask[:, t]
                + (1 - response_mask[:, t]) * lastgaelam
            )

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (
                    id2std[index[i]] + epsilon
                )
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(
    AdvantageEstimator.GRPO_PASSK
)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[
                    index[i]
                ] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS
)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (
            (token_level_rewards * response_mask)
            .flip(dims=[-1])
            .cumsum(dim=-1)
            .flip(dims=[-1])
        )
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(
            loss_mask, dim=-1
        )  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # """
    # Compute the clipped policy objective and related metrics for PPO.

    # Adapted from
    # https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    # Args:
    #     old_log_prob (torch.Tensor):
    #         Log-probabilities of actions under the old policy, shape (batch_size, response_length).
    #     log_prob (torch.Tensor):
    #         Log-probabilities of actions under the current policy, shape (batch_size, response_length).
    #     advantages (torch.Tensor):
    #         Advantage estimates for each action, shape (batch_size, response_length).
    #     response_mask (torch.Tensor):
    #         Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
    #     loss_agg_mode (str, optional):
    #         Aggregation mode for `agg_loss`. Defaults to "token-mean".
    # """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = (
        config.clip_ratio
    )  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = (
        config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    )
    clip_ratio_high = (
        config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    )
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )
    # add more metircs:
    neg_valid = ratio[(advantages < 0) & response_mask.bool()]
    if neg_valid.numel() > 0:
        neg_is_min = neg_valid.min()
        neg_is_max = neg_valid.max()
        neg_is_p75 = torch.quantile(neg_valid, 0.75)
        neg_is_p995 = torch.quantile(neg_valid, 0.995)
        neg_is_p999 = torch.quantile(neg_valid, 0.999)
    else:
        neg_is_min = torch.tensor(0.0, device=ratio.device)
        neg_is_max = torch.tensor(0.0, device=ratio.device)
        neg_is_p995 = torch.tensor(0.0, device=ratio.device)
        neg_is_p999 = torch.tensor(0.0, device=ratio.device)
        neg_is_p75 = torch.tensor(0.0, device=ratio.device)

    pos_valid = ratio[(advantages > 0) & response_mask.bool()]
    if pos_valid.numel() > 0:
        pos_is_max = pos_valid.max()
        pos_is_p25 = torch.quantile(pos_valid, 0.25)
        pos_is_median = torch.quantile(pos_valid, 0.5)
        pos_is_p75 = torch.quantile(pos_valid, 0.75)
        pos_is_p995 = torch.quantile(pos_valid, 0.995)
        pos_is_p999 = torch.quantile(pos_valid, 0.999)
        pos_is_min = pos_valid.min()
    else:
        pos_is_p25 = torch.tensor(0.0, device=ratio.device)
        pos_is_max = torch.tensor(0.0, device=ratio.device)
        pos_is_median = torch.tensor(0.0, device=ratio.device)
        pos_is_p75 = torch.tensor(0.0, device=ratio.device)
        pos_is_p995 = torch.tensor(0.0, device=ratio.device)
        pos_is_p995 = torch.tensor(0.0, device=ratio.device)
        pos_is_p999 = torch.tensor(0.0, device=ratio.device)
        pos_is_min = torch.tensor(0.0, device=ratio.device)

    pg_metrics = {
        "actor/neg_is_max": neg_is_max.detach().item(),
        "actor/neg_is_min": neg_is_min.detach().item(),
        "actor/neg_is_p995": neg_is_p995.detach().item(),
        "actor/neg_is_p999": neg_is_p999.detach().item(),
        "actor/neg_is_p75": neg_is_p75.detach().item(),
        "actor/pos_is_max": pos_is_max.detach().item(),
        "actor/pos_is_median": pos_is_median.detach().item(),
        "actor/pos_is_p75": pos_is_p75.detach().item(),
        "actor/pos_is_p995": pos_is_p995.detach().item(),
        "actor/pos_is_p999": pos_is_p999.detach().item(),
        "actor/pos_is_min": pos_is_min.detach().item(),
        "actor/pos_is_p25": pos_is_p25.detach().item(),
    }

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, pg_metrics


"""
FIPO的损失策略实现：

该函数的主要目的是计算 policy_loss，通常包含两个核心部分：
1-PPO Clipped Objective（PPO 截断目标）：防止策略更新过大。
2-KL Penalty（KL 散度惩罚）：防止策略偏离参考模型太远（即网页解析中提到的 FutureKL 逻辑）
"""


@register_policy_loss("future_kl")
def compute_policy_loss_future_kl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    old_log_prob	(Batch, Seq)	采样时的对数概率（旧策略），用于计算比率，不参与反向传播。
    log_prob	(Batch, Seq)	当前策略的对数概率，用于计算比率和 KL 散度。
    advantages	(Batch, Seq)	优势函数，衡量某个动作比平均表现好多少。
    response_mask	(Batch, Seq)	掩码，用于过滤掉 Prompt 部分，只计算回答（Response）的 Loss。
    loss_agg_mode	str	聚合方式，通常为 "token-mean"（按有效 token 数取平均）或 "batch-mean"。

    """

    """
    确保传入的配置对象有效。
    含义：代码强制要求 config 必须存在，且不能是某种特定的旧类型（AlgoConfig），这通常是为了保证配置的格式符合当前函数的预期（例如必须是 DictConfig）


    """
    assert config is not None
    assert not isinstance(config, AlgoConfig)

    """
    
    """
    clip_ratio = (
        config.clip_ratio
    )  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = (
        config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    )
    clip_ratio_high = (
        config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    )

    """
    在标准 PPO 中，当优势函数（Advantage）为负数且比率（Ratio）非常小（小于 1−ϵ ）时，损失函数会变成常数，导致梯度消失，模型无法从“极差的动作”中学习。
    Dual-Clip 机制：
        - 论文 [1] 提出了 Dual-Clip PPO 来解决这个问题。
        - 当 Advantage < 0 且 Ratio 非常小时，引入一个下界常数 c （即代码中的 clip_ratio_c）。
        - 断言检查：代码强制要求 clip_ratio_c > 1.0，这是 Dual-Clip 算法的数学要求。
    """
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    """
    计算对数比率 (negative_approx_kl)
    """
    negative_approx_kl = log_prob - old_log_prob

    """
    防止数值溢出或下溢。
        原因：
            - 如果 log_prob 和 old_log_prob 差异过大（例如新策略觉得某个词概率极高，旧策略觉得极低），直接计算指数 exp(x) 可能会导致结果变成 inf（无穷大）或 0。
            - 将值限制在 [-20, 20] 之间，可以确保后续的 exp() 操作在浮点数的安全范围内，防止梯度爆炸或消失。
    """
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    """
    ratio > 1：新策略更喜欢这个动作。
    ratio < 1：新策略更不喜欢这个动作。
    """
    ratio = torch.exp(negative_approx_kl)
    """
    masked_mean 的作用：
        response_mask 用于过滤掉 Prompt（提示词）部分，只计算生成内容（Response）的 KL 散度。
    用途：
        这个值通常不直接参与梯度反向传播，而是作为一个监控指标，用来观察模型训练过程中是否偏离参考模型太远。
    """
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # let's compute the future kl, which is the kl accumulated from the current token to the end of the response(within the response_mask)

    """
    防御性编程。
    目的：确保输入的张量形状一致。
        - 如果 log_prob 和 advantages 形状不匹配，后续的乘法运算会报错或产生错误的广播结果。
        - 确保 response_mask 的批次大小（Batch Size）与数据一致，防止掩码错位导致计算了错误的 token。
    """
    assert (
        log_prob.shape == old_log_prob.shape == advantages.shape
    ), f"log/old/adv shape mismatch: {log_prob.shape}, {old_log_prob.shape}, {advantages.shape}"

    assert response_mask.dim() == 2 and response_mask.size(0) == log_prob.size(
        0
    ), f"response_mask shape {response_mask.shape} incompatible with batch {log_prob.shape}"

    """
    针对“坏动作”（优势为负），计算一个“未来累积的 KL 散度”作为惩罚或筛选条件。
    简单来说，它在问：“如果一个动作不好（Advantage < 0），而且它导致模型偏离参考模型太远（KL 很大），并且这种偏离在未来还会持续，那我们就应该重点惩罚或忽略它。


    """
    # calculate future_kl using negative_approx_kl and response_mask
    batch_size, response_len = log_prob.shape
    device = log_prob.device

    assert (
        response_mask.size(1) == response_len
    ), f"Time dim mismatch: log_prob length={response_len}, response_mask length={response_mask.size(1)}"

    chunk_size = config.policy_loss.get("chunk_size", 128)
    decay_rate = config.policy_loss.get("decay_rate", 128)
    """
    gamma 计算：
            这里计算了一个衰减因子γ 。
            公式 2 ** (−1/128)
            意味着每 128 个 token，权重减半。这通常用于远期信用（Future Credit）的分配，
            即越靠近当前的时刻，对未来的影响权重越大。
    """
    gamma = 2 ** (-1.0 / decay_rate)

    """
    初始化一个全零张量，准备用来存储每个时刻对应的“未来累积 KL”
    """
    future_kl = torch.zeros(
        (batch_size, response_len), device=device, dtype=log_prob.dtype
    )
    pos_i = torch.arange(response_len, device=device).unsqueeze(1)  # (L,1)
    # to avoid high is token from the sample to deviate our training and weighting
    # we exclude those greater than clip_frac_c. These tokens have no gradient neither in the following training.

    """

    核心参数 clip_ratio_c：
        - 这是 Dual-Clip PPO（参考资料 [1], [5]）特有的参数。
        - 在标准 PPO 中，当优势 A<0 且比率 r<1−ϵ 时，梯度会消失。Dual-Clip 引入了一个下界c （即 clip_ratio_c），通常设为 3.0 或更大。

    对数变换：
        因为前面的 negative_approx_kl 是 log(r) ，所以这里也要对 clip_ratio_c 取对数，以便在同一维度比较。
    """
    filter_threshold = torch.log(
        torch.tensor(clip_ratio_c, device=device, dtype=log_prob.dtype)
    )

    """
        ignore_mask：标记那些 log(r)>log(c) 的位置，即 r>c 。
        含义：如果一个动作的比率 r 非常大（超过了 clip_ratio_c），说明新策略极其不喜欢这个动作（概率极低）。
             在 Dual-Clip 理论中，这种极端情况下的梯度是不稳定的或无意义的，因此将其标记为“忽略”。
        participation_mask：取反后，表示“有效的、需要计算梯度的 token”。
    """
    is_negative_adv = advantages < 0  # .view(batch_size, 1) # bsz, L
    ignore_mask = negative_approx_kl > filter_threshold  # bsz, L
    participation_mask = ~ignore_mask

    """
    1. 先用 response_mask 过滤掉 Prompt，只保留生成内容的 KL 值
    2. 再乘以 participation_mask。
    3. 最终效果：kl_response 现在只包含那些既在生成内容中，又没有超过 Dual-Clip 阈值的 KL 散度值。

    """
    kl_response_premask = negative_approx_kl * response_mask.to(
        log_prob.dtype
    )  # response mased kl diff
    kl_response = kl_response_premask * participation_mask.to(log_prob.dtype)

    """
    
    这段代码通过分块矩阵乘法（Chunked Matrix Multiplication）的方式，高效地计算了未来 KL 散度（Future KL Divergence）的累积值。

    它解决了传统循环计算慢的问题，利用 GPU 的并行计算能力，一次性算出序列中每个时刻对未来的累积影响。


    """
    gamma_t = torch.tensor(gamma, dtype=log_prob.dtype, device=device)
    """
    目的：为了节省显存。如果序列很长（例如 2048），直接构建一个 2048×2048 的矩阵可能会爆显存。
    做法：将序列切分成小块（例如每块 128 个 token），逐块计算贡献，最后累加。
    """
    for j_start in range(0, response_len, chunk_size):
        j_end = min(response_len, j_start + chunk_size)
        j_idx = torch.arange(j_start, j_end, device=device).unsqueeze(0)  # (1, Kb)

        """
        pos_i：形状为 (L,1) ，代表当前时刻 t （行索引）。
        j_idx：形状为 (1,K_b) ，代表未来时刻 k （列索引，当前块内的）。
        distance：通过广播机制相减，得到一个 (L,K b) 的矩阵。矩阵中的每个元素 (i,k) 表示时间差 k−i 。
        mask：只保留 k≥i 的部分（即只看未来，不看过去），将过去的时间步掩码掉。
        """
        # distance shape (L, Kb) where entry (i,k) = j - i
        distance = j_idx - pos_i
        mask = distance >= 0  # zero out j < i
        distance_clamped = distance.clamp(min=0)
        """
        当 k=i （当前时刻），距离为 0，权重为 γ**0=1

        当 k=i+1 （下一时刻），权重为 γ**1

        当 k<i （过去时刻），由于 mask，权重为 0

        结果：decay_block 是一个下三角矩阵（在当前块范围内），存储了所有的时间衰减系数。
        """
        # decay_block (L, Kb)
        decay_block = torch.pow(gamma_t, distance_clamped) * mask.to(log_prob.dtype)

        """
        kl_block：提取当前块内的 KL 值，形状 (B, K_b)。
        decay_block.t()：转置后形状变为 (K_b, L)。
        matmul 这是一个加权求和操作

        累加：将所有块的贡献 contrib 加到 future_kl 中，最终得到完整的未来累积 KL
        """
        # kl_block (B, Kb)
        kl_block = kl_response[:, j_start:j_end]
        # contribution: for this block, contrib_{b,i} = sum_k kl_block[b,k] * decay_block[i,k]
        # compute via matmul: (B, Kb) @ (Kb, L) -> (B, L)
        contrib = torch.matmul(kl_block, decay_block.t())
        future_kl += contrib

    """
    Future KL 机制的“执行阶段”。它根据配置，利用之前计算好的 future_kl（未来累积 KL 散度），生成一个影响力权重（Influence Weights），用来调节策略更新的幅度。
        简单来说，这段代码在回答一个问题：“当我们发现某个动作会导致未来偏离参考模型太远时，我们应该如何调整当前的更新力度？”

    这段代码是在动态调整学习率。
        1.它先计算了每个 token 对未来造成的潜在 KL 散度（future_kl）。
        2.然后，它根据配置策略，把这个 KL 值转换成一个权重系数（influence_weights）。
        3.最后（在后续代码中），这个系数会乘在 PPO 的 Loss 上。
    """
    if config.policy_loss.get("future_kl_clip_ratio") != 0.0:
        clip_ratio = config.policy_loss.get("future_kl_clip_ratio")
        if not config.policy_loss.get("future_kl_clip_high_only"):
            """
            标准对称裁剪
                既防止权重过大（导致训练不稳定），也防止权重过小（导致梯度消失）。
                适用场景：代码注释提到“works well with smaller models such as 7b”。小模型通常熵较低，比较脆弱，这种温和的对称约束能保护模型不发生剧烈震荡。
            """
            # seems to work well with smaller models such as 7b --> usually create lower entropy
            upper_bound = 1.0 + clip_ratio
            lower_bound = 1.0 - clip_ratio
            """
            本质：这是一个缩放系数（Scaling Factor）。
            来源：它是 future_kl 的指数形式 exp(future_kl) 。
                - 回顾之前的计算，future_kl 本质上是 log(ratio) 的累积。
                - 所以 exp(future_kl) 近似于一个“累积比率（Cumulative Ratio）”。
            作用：这个权重最终会乘在 PPO 的损失函数（Loss）上。
                - 如果权重 大：说明这个动作对未来影响大且符合预期，鼓励更新。
                - 如果权重 小：说明这个动作可能导致未来失控，抑制更新。
            .detach()：非常重要。这个权重只是用来调节梯度的“阀门”，它本身不参与反向传播（不需要计算梯度的梯度）
            """
            influence_weights = torch.clamp(
                torch.exp(future_kl), min=lower_bound, max=upper_bound
            ).detach()
        else:
            """
            仅高位裁剪:
            逻辑：将权重限制在 [1.0,1+ϵ] 之间。
            含义：这是一种激进（Radical）的策略。
                - 只允许放大，不允许缩小：它保证了 influence_weights 至少是 1.0。这意味着只要计算出了 Future KL，我们至少会保持原有的更新力度，甚至加大它，但绝不会减弱它。
                - 目的：代码注释提到“works fine for larger model to break boundary”。大模型往往容易陷入局部最优或“死板”，这种策略强制模型保持探索力度，防止 Future KL 的惩罚把模型“压垮”或导致梯度消失。
            """
            # a radical way to update model, works fine for larger model to break boundary hopefully
            upper_bound = 1.0 + clip_ratio
            lower_bound = 1.0

            influence_weights = torch.clamp(
                torch.exp(future_kl), min=1.0, max=1.0 + clip_ratio
            ).detach()
    else:
        """
        无裁剪/仅上限:
        逻辑：几乎不限制下限，只限制一个极大的上限（10.0）防止溢出。
        含义：
            - 这是最宽松的模式。它允许 future_kl 自由地调节权重，哪怕变得很小（接近 0）。
            - 适用场景：通常用于调试，或者当你希望 Future KL 机制完全自由地发挥作用，不进行人工干预时。

        """
        upper_bound = 10.0
        lower_bound = 0.0
        influence_weights = torch.clamp(torch.exp(future_kl), max=10.0).detach()

    """
    这段代码是 Dual-Clip PPO 算法的核心计算部分，它结合了 Future KL 机制来生成最终的策略梯度损失。
    这段代码主要完成了三件事：
        - 安全保护：防止模型对“坏动作”过度惩罚。
        - 加权优势函数：利用之前计算的 influence_weights 调节学习力度。
        - Dual-Clip 损失计算：计算并合并 PPO-Clip 和 Dual-Clip 的损失。

    """
    # Apply a safety threshold: if a negative sample's IS value is too high and its weight is increasing, cap it at the baseline value (1.0)
    # To avoid over-penalization

    """
    安全阈值保护:

    背景：
        advantages < 0：表示这是一个坏动作（比平均水平差）。
        ratio > safe_threshold：表示新策略下这个坏动作的概率异常高（比如 >4.0 ）。这通常发生在模型“学坏了”或者策略崩塌时。
    问题：如果直接应用 influence_weights（可能很大），会导致 Loss 变得极大，梯度爆炸，或者过度惩罚导致模型参数剧烈震荡。
    解决方案：
            - 检测到这种情况时，强制将 influence_weights 限制在 [0.8,1.0] 之间。
            - 含义：“虽然这个动作很烂且概率很高，但我们不要过度反应，温和地把它压下去就好。”

    这段代码实现了一个带有 Future KL 调节的 Dual-Clip PPO 损失函数。
    1.先用 influence_weights 根据未来的 KL 散度调节当前的 Advantage。
    2.再用标准 PPO-Clip 逻辑计算主要损失。
    3.最后用 Dual-Clip 逻辑修补“极差动作”的梯度消失问题。
    4.全程包含安全检查，防止数值异常导致训练崩溃。
    """
    safe_threshold = config.policy_loss.get("safety_thresh", 4.0)
    mask_neg_high_is = (advantages < 0) & (ratio > safe_threshold)
    influence_weights = torch.where(
        mask_neg_high_is,
        torch.clamp(influence_weights, min=0.8, max=1.0),
        influence_weights,
    )

    """
    统计监控:
        作用：这部分代码不参与梯度计算，纯粹是为了记录日志（Logging）。
        监控指标：
            clip_frac：有多少比例的数据被裁剪了（触达了上下界）。
            influence_weights_mean：调节系数的平均值，用来观察 Future KL 机制是在“加速”还是“减速”训练。
    """
    # calcuate clip ratio
    clip_frac_upper = verl_F.masked_mean(
        (influence_weights >= upper_bound - 1e-7).float(), response_mask
    )
    clip_frac_lower = verl_F.masked_mean(
        (influence_weights <= lower_bound + 1e-7).float(), response_mask
    )
    total_clip_frac = clip_frac_upper + clip_frac_lower
    # add stats for raw influence weight
    influence_weights_mean_raw = verl_F.masked_mean(torch.exp(future_kl), response_mask)
    valid_vals_raw = torch.exp(future_kl)[
        response_mask.to(dtype=torch.bool, device=influence_weights.device)
    ]
    raw_influence_weights_min = valid_vals_raw.min()
    raw_influence_weights_max = valid_vals_raw.max()
    # add status check of the influence_weights
    influence_weights_mean = verl_F.masked_mean(influence_weights, response_mask)
    # influence_weights_std = verl_F.masked_std(influence_weights, response_mask)
    valid_vals = influence_weights[
        response_mask.to(dtype=torch.bool, device=influence_weights.device)
    ]
    influence_weights_min = valid_vals.min()
    influence_weights_max = valid_vals.max()

    """
    计算加权优势:
    核心逻辑：这是 Future KL 机制生效的地方。
    含义：
        - 原来的 PPO 只看 advantages（动作好不好）。
        - 现在的 PPO 看 weighted_advantages（动作好不好 × 对未来影响大不大）。
        - 如果某个动作会导致未来严重偏离（influence_weights 小），即使它当前奖励还行，也会被抑制。

    """
    weighted_advantages = advantages * influence_weights

    """
    clip_pg_losses1 (Final PPO Loss)：取最大值。
        当 Advantage > 0：取 min 还是 max？
            - 标准 PPO 是 min(ratio * A, clip * A)。
            - 这里代码写的是 maximum，且前面加了负号。
            - 数学上： −min(x,y)=max(−x,−y) 。
            - 所以逻辑是一致的：防止优势被过度放大。
        当 Advantage < 0：
            - 标准 PPO 是 max(ratio * A, clip * A)（因为 A 是负数，max 意味着取绝对值小的那个，即防止过度惩罚）。
            - 这里同样通过 maximum 和负号实现了这一逻辑。
    """
    pg_losses1 = -weighted_advantages * ratio  # 未裁剪的损失
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -weighted_advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A  裁剪后的损失。

    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )

    """
    Dual-Clip 损失:
        背景：这是针对 Advantage < 0 且 Ratio 极小（ <1−ϵ ）的情况。
    问题：在标准 PPO 中，如果 Ratio 极小，clip_pg_losses1 会变成常数（因为被 clamp 卡住了），导致梯度为 0，模型无法从这个“极差的动作”中学习。
    Dual-Clip 解决方案：
        - 引入 clip_ratio_c（通常 >1 ）。
        - pg_losses3：计算基于下界 c 的损失。
        - torch.min(...)：
            - 因为 Advantage 是负数，-weighted_advantages 是正数。
            - 这里实际上是在限制 Loss 的上界，确保即使在极端情况下，也能保留一个非零的梯度（由 c 决定），让模型知道“这个动作真的很差，要赶紧改”。
    """

    pg_losses3 = -weighted_advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    """
    这段代码是训练流程的最后一步，主要完成了三个关键任务：数据过滤、损失合并以及最终聚合。
    它像一个质量检查员，在计算最终损失前，把那些“学坏了”的数据剔除掉，确保模型只从高质量的样本中学习

    
    """
    # Stats info to collect：
    # raw influence weight，lower clip token count, done
    # Percentages of IS negative samples > 2, 3, 4, 10
    # Negative sample is: max，995 percent，999 percent, done
    # Positive sample is: max，995 percent，999 percent, done
    # Extremely small is from Positive sample (could result in large distribution shift)

    # filter mechanism： if a sequence contains more than 1 token that has been clip by dual clip, then we throw away the entire sequence.
    # however, this is rarely activated in the 32b training.
    """
    序列级过滤机制:
    含义：这是一个非常严格的筛选条件，用来标记那些“极度糟糕且被 Dual-Clip 强行修正”的 Token。
        - advantages < 0：动作很烂。
        - clip_pg_losses1 > pg_losses3：标准 PPO 的裁剪损失（clip_pg_losses1）比 Dual-Clip 的下界损失（pg_losses3）还要大。这意味着这个动作差到了极点，以至于触发了 Dual-Clip 的“兜底机制”（即代码注释中的 "clip by dual clip"）。
    目的：找出那些模型不仅做错了，而且错得非常离谱，需要算法强行介入修正的 Token。
    """
    lower_clip_mask = (
        (advantages < 0) & (clip_pg_losses1 > pg_losses3) & response_mask.bool()
    )

    """
    逻辑：
        - 统计每个序列（Sequence）中有多少个这样的“极度糟糕 Token”。
        - 阈值判定：如果一个序列中有 超过 1 个 这样的 Token，就认为这个序列彻底“坏掉”了（Corrupted）。
    注释解读：代码注释提到 this is rarely activated in the 32b training。说明在大规模模型训练中，模型通常比较稳定，很少会出现连续生成多个“极度离谱”动作的情况。
    """
    low_clip_token_counts = lower_clip_mask.sum(dim=1)  # (batch,）

    # sequence-level: whether this entire response should be invalidated
    seq_has_low_clip = (
        low_clip_token_counts > 1
    )  # (batch,) # hard threshold (if sequence has many, > threshold,--> sequence)

    """
    结果：
        - 如果序列被判定为“坏掉”，seq_valid_mask 为 False，整个序列的 final_mask 都会变成 0。
        - 效果：在计算 Loss 时，整个序列的梯度都会被丢弃。这就像老师改卷子，发现这道题不仅做错了，而且错得离谱且毫无逻辑，直接整张卷子作废，不让学生从这个样本中学习。

    """
    seq_valid_mask = (~seq_has_low_clip).unsqueeze(1)  # (batch,1)

    final_mask = response_mask.bool() & seq_valid_mask  # (batch, response_len)
    final_mask_f = final_mask.to(log_prob.dtype)

    """
    损失合并:

    核心逻辑：根据 Advantage 的正负，选择不同的损失计算结果。
        当 weighted_advantages >= 0 (好动作)：
            - 使用 clip_pg_losses1（标准 PPO-Clip 损失）。
            - 逻辑：对于好动作，我们只需要标准的 PPO 裁剪机制来防止过度优化即可。
        当 weighted_advantages < 0 (坏动作)：
            - 使用 clip_pg_losses2（Dual-Clip 损失）。
            - 逻辑：对于坏动作，我们需要启用 Dual-Clip 机制（即之前计算的 pg_losses3 相关的逻辑），以防止在 Ratio 极小时梯度消失。
    """
    pg_losses = torch.where(weighted_advantages < 0, clip_pg_losses2, clip_pg_losses1)

    """
    最终聚合:

    作用：将处理好的 Loss 矩阵压缩成一个标量。
    final_mask_f：应用了之前所有的过滤规则（Response 掩码 + 坏序列剔除）。
    loss_agg_mode：
        - 通常是 "token-mean"（对所有有效 Token 取平均）或 "batch-mean"。
        - 这一步决定了最终反向传播的梯度大小。
    """
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=final_mask_f, loss_agg_mode=loss_agg_mode
    )

    """
    这段代码是训练过程中的“体检报告生成器”。它不参与梯度更新，而是专注于收集关于 重要性采样比率（Ratio） 的统计信息。

    在 PPO 训练中，ratio 是衡量新旧策略差异的核心指标。这段代码通过监控 ratio 在“好动作”和“坏动作”下的分布情况，
    帮助开发者判断模型是否训练稳定、是否存在梯度消失或策略崩塌的风险

     监控“坏动作”的偏离程度:
    """

    """
    背景：is_negative_adv 表示这是一个坏动作（Advantage < 0）。理论上，我们希望新策略降低这个动作的概率，即 ratio 应该很小（接近 0）。
    异常检测：
        - 如果 ratio >= 2.0 且是坏动作，说明新策略反而把这个坏动作的概率提高了 2 倍以上。这是非常危险的信号，说明模型正在“学坏”。
        - 代码统计了 ratio 在 [2, 3), [3, 4), [4, clip_ratio_c) 区间的比例。
    作用：这些指标如果过高，说明训练非常不稳定，或者学习率太大，导致模型在优化坏动作时发生了震荡。
    """
    # Start gathering the rest of stats information
    neg_ratio_2_3 = verl_F.masked_mean(
        ((ratio >= 2.0) & (ratio < 3.0) & is_negative_adv).float(), response_mask
    )
    neg_ratio_3_4 = verl_F.masked_mean(
        ((ratio >= 3.0) & (ratio < 4.0) & is_negative_adv).float(), response_mask
    )
    neg_ratio_4_10 = verl_F.masked_mean(
        ((ratio >= 4.0) & (ratio < clip_ratio_c) & is_negative_adv).float(),
        response_mask,
    )

    """
    “坏动作”的分位数统计:

    逻辑：提取所有坏动作对应的 ratio，计算最大值和高分位点（75%, 99.5%, 99.9%）。
    目的：
        - neg_is_max：查看最极端的“学坏”案例有多严重。
        - neg_is_p995：排除掉极少数异常值后，观察绝大多数“坏动作”的 ratio 上限。如果这个值远大于 1.0，说明模型普遍在增加坏动作的概率，训练方向可能反了。

    """
    neg_valid = ratio[(advantages < 0) & response_mask.bool()]
    if neg_valid.numel() > 0:
        neg_is_max = neg_valid.max()
        neg_is_p75 = torch.quantile(neg_valid, 0.75)
        neg_is_p995 = torch.quantile(neg_valid, 0.995)
        neg_is_p999 = torch.quantile(neg_valid, 0.999)
    else:
        neg_is_max = torch.tensor(0.0, device=ratio.device)
        neg_is_p995 = torch.tensor(0.0, device=ratio.device)
        neg_is_p999 = torch.tensor(0.0, device=ratio.device)
        neg_is_p75 = torch.tensor(0.0, device=ratio.device)

    """
    背景：advantages > 0 表示好动作。理论上我们希望 ratio 变大（> 1.0）。
    关键指标：
        pos_is_max / pos_is_p999：监控好动作的 ratio 是否过大。如果过大（例如 > 10.0），说明策略更新过猛，
                                  容易导致“模式崩塌”（Model Collapse），即模型只生成某一种安全但重复的内容。
        pos_is_min：监控好动作的 ratio 是否过小。
    """
    pos_valid = ratio[(advantages > 0) & response_mask.bool()]
    if pos_valid.numel() > 0:
        pos_is_max = pos_valid.max()
        pos_is_p25 = torch.quantile(pos_valid, 0.25)
        pos_is_median = torch.quantile(pos_valid, 0.5)
        pos_is_p75 = torch.quantile(pos_valid, 0.75)
        pos_is_p995 = torch.quantile(pos_valid, 0.995)
        pos_is_p999 = torch.quantile(pos_valid, 0.999)
        pos_is_min = pos_valid.min()
    else:
        pos_is_p25 = torch.tensor(0.0, device=ratio.device)
        pos_is_max = torch.tensor(0.0, device=ratio.device)
        pos_is_median = torch.tensor(0.0, device=ratio.device)
        pos_is_p75 = torch.tensor(0.0, device=ratio.device)
        pos_is_p995 = torch.tensor(0.0, device=ratio.device)
        pos_is_p995 = torch.tensor(0.0, device=ratio.device)
        pos_is_p999 = torch.tensor(0.0, device=ratio.device)
        pos_is_min = torch.tensor(0.0, device=ratio.device)

    """
    梯度消失预警:

    含义：统计那些明明是“好动作”（Advantage > 0），但 ratio 却极小（< 0.001）的 Token 比例。
    严重性：
        - 当 ratio 极小时， log(ratio) 会非常负，且梯度极其微弱。
        - 这意味着模型几乎完全抛弃了这些原本应该被鼓励的好动作。
        - 如果 pos_mini_frac 很高，说明发生了严重的梯度消失或策略偏离，模型可能已经“忘记”了如何生成好的回答，或者陷入了局部最优。
    """
    pos_mini_frac = verl_F.masked_mean(
        ((ratio < 1e-3) & (advantages > 0)).float(), response_mask
    )

    return (
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
        kl_response_premask,
    )


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = (
        config.clip_ratio_low
        if config.clip_ratio_low is not None
        else config.clip_ratio
    )
    clip_ratio_high = (
        config.clip_ratio_high
        if config.clip_ratio_high is not None
        else config.clip_ratio
    )

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = (
        torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths
    )

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = (
        log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    )
    log_seq_importance_ratio = torch.clamp(
        log_seq_importance_ratio, max=10.0
    )  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(
        seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high
    )
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean"
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
):
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    pg_losses = -log_prob * advantages

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )
    return pg_loss, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = (
        config.policy_loss.clip_cov_ratio
        if config.policy_loss.clip_cov_ratio is not None
        else 0.0002
    )
    cliprange = config.clip_ratio
    cliprange_low = (
        config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    )
    cliprange_high = (
        config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    )
    clip_cov_ub = (
        config.policy_loss.clip_cov_ub
        if config.policy_loss.clip_cov_ub is not None
        else 5.0
    )
    clip_cov_lb = (
        config.policy_loss.clip_cov_lb
        if config.policy_loss.clip_cov_lb is not None
        else 1.0
    )

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.0)


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = (
        config.policy_loss.kl_cov_ratio
        if config.policy_loss.kl_cov_ratio is not None
        else 0.0002
    )
    ppo_kl_coef = (
        config.policy_loss.ppo_kl_coef
        if config.policy_loss.ppo_kl_coef is not None
        else 1.0
    )

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (
            all_valid_logp - all_valid_logp.mean()
        )
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[
                large_cov_idxs // advantages.shape[1],
                large_cov_idxs % advantages.shape[1],
            ] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1],
                large_cov_idxs % advantages.shape[1],
            ]

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    return pg_loss, torch.tensor(0.0), ppo_kl_abs, torch.tensor(0.0)


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = (
        config.clip_ratio
    )  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = (
        config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    )
    clip_ratio_high = (
        config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(
        negative_approx_kl, -cliprange_low, cliprange_high
    )
    negative_approx_kl_min = torch.min(
        sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp
    )
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp(
        (negative_approx_kl_min * response_mask).sum(dim=-1)
        / (response_mask_sum + 1e-8)
    )
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio
    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean(
        (clipped * (advantages > 0)).float(), response_mask
    )
    pg_clipfrac_lower = verl_F.masked_mean(
        (clipped * (advantages < 0)).float(), response_mask
    )

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(
        loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(
        vpreds, values - cliprange_value, values + cliprange_value
    )
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(
        loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )
    vf_clipfrac = verl_F.masked_mean(
        torch.gt(vf_losses2, vf_losses1).float(), response_mask
    )
    return vf_loss, vf_clipfrac


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(
        scores: torch.Tensor, reweight_method: str, weight_pow: float
    ) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where(
                (scores == max_score) | (scores == min_score), 1.0, 0.0
            )
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {
        key: tensor[sample_indices] for key, tensor in data.batch.items()
    }

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
