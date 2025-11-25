from typing import Tuple

import torch
import torch.nn.functional as F


def ppo_objective(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float,
) -> torch.Tensor:
    ratio = torch.exp(new_logp - old_logp)
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    obj = torch.min(ratio * advantages, clipped)
    return -obj.mean()


def value_loss(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(values, targets)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-9)
    return -(probs * log_probs).sum(dim=-1).mean()


def update_rcpo_lambda(lmbda: torch.Tensor, quality_loss: torch.Tensor, epsilon: float, lr: float):
    with torch.no_grad():
        violation = quality_loss.detach() - epsilon
        new_lambda = torch.clamp(lmbda + lr * violation, min=0.0)
        lmbda.copy_(new_lambda)
    return lmbda


def compute_advantages(rewards: torch.Tensor, values: torch.Tensor, gamma: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    returns = rewards * gamma
    adv = returns - values.detach()
    return adv, returns

