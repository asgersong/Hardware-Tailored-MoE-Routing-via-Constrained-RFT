from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class VirtualClusterConfig:
    num_experts: int = 8
    num_nodes: int = 2
    intra_node_latency: float = 0.05
    inter_node_latency: float = 2.5
    compute_cost: float = 1.0
    load_penalty_gamma: float = 0.001


class VirtualClusterEnv:
    """
    Lightweight latency simulator used as the reward model.
    """

    def __init__(self, config: Optional[VirtualClusterConfig] = None):
        self.config = config or VirtualClusterConfig()
        self.expert_to_node = torch.tensor(
            [i // (self.config.num_experts // self.config.num_nodes) for i in range(self.config.num_experts)],
            dtype=torch.long,
        )

    def compute_latency(self, token_nodes: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_nodes: (batch*seq,) virtual source node id for each token.
            selected_experts: (batch*seq, k) indices of chosen experts.
        Returns:
            latency per token (batch*seq,)
        """
        device = selected_experts.device
        token_nodes = token_nodes.to(device)
        expert_nodes = self.expert_to_node.to(device)[selected_experts]
        is_remote = expert_nodes != token_nodes.unsqueeze(-1)
        comm_cost = torch.where(
            is_remote,
            torch.tensor(self.config.inter_node_latency, device=device),
            torch.tensor(self.config.intra_node_latency, device=device),
        ).sum(dim=-1)

        flat_experts = selected_experts.reshape(-1)
        counts = torch.bincount(flat_experts, minlength=self.config.num_experts).float()
        load_penalty = torch.zeros_like(token_nodes, dtype=torch.float, device=device)
        if self.config.load_penalty_gamma > 0:
            penalty_per_expert = self.config.load_penalty_gamma * (counts ** 2)
            load_penalty = penalty_per_expert[flat_experts].view_as(selected_experts).mean(dim=-1)

        latency = (
            torch.tensor(self.config.compute_cost, device=device) + comm_cost + load_penalty
        )
        return latency

    def reward(self, token_nodes: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        return -self.compute_latency(token_nodes, selected_experts)

