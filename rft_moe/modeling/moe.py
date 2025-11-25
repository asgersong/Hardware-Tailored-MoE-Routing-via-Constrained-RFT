from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouterOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    selected_experts: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    hidden: torch.Tensor


class ExpertMLP(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim, bias=bias)
        self.fc2 = nn.Linear(4 * dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MoELayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        k: int = 2,
        router_init_std: float = 0.02,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = k
        self.router = nn.Linear(dim, num_experts, bias=bias)
        self.value_head = nn.Linear(dim, 1, bias=True)
        self.experts = nn.ModuleList(
            [ExpertMLP(dim, dropout=dropout, bias=bias) for _ in range(num_experts)]
        )
        nn.init.normal_(self.router.weight, mean=0.0, std=router_init_std)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=router_init_std)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)

    def forward(
        self, x: torch.Tensor, return_router_outputs: bool = False
    ) -> Tuple[torch.Tensor, Optional[RouterOutput]]:
        batch, seq, dim = x.shape
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=self.k, dim=-1)
        log_prob = torch.log(topk_vals + 1e-9).sum(dim=-1)

        x_flat = x.reshape(batch * seq, dim)
        topk_idx_flat = topk_idx.reshape(batch * seq, self.k)
        topk_vals_flat = topk_vals.reshape(batch * seq, self.k)
        outputs = torch.zeros_like(x_flat)

        for expert_id, expert in enumerate(self.experts):
            for slot in range(self.k):
                slot_indices = topk_idx_flat[:, slot]
                slot_weights = topk_vals_flat[:, slot]
                slot_mask = slot_indices == expert_id
                if slot_mask.sum() == 0:
                    continue
                expert_inp = x_flat[slot_mask]
                routed = expert(expert_inp)
                weight = slot_weights[slot_mask].unsqueeze(-1)
                outputs[slot_mask] += weight * routed

        outputs = outputs.view(batch, seq, dim)

        router_output = None
        if return_router_outputs:
            value = self.value_head(x).squeeze(-1)
            router_output = RouterOutput(
                logits=logits,
                probs=probs,
                selected_experts=topk_idx,
                log_prob=log_prob,
                value=value,
                hidden=x.detach(),
            )
        return outputs, router_output

    def load_from_dense(self, dense_ffn: nn.Module):
        with torch.no_grad():
            for expert in self.experts:
                expert.fc1.weight.copy_(dense_ffn.c_fc.weight)
                if expert.fc1.bias is not None and dense_ffn.c_fc.bias is not None:
                    expert.fc1.bias.copy_(dense_ffn.c_fc.bias)
                expert.fc2.weight.copy_(dense_ffn.c_proj.weight)
                if expert.fc2.bias is not None and dense_ffn.c_proj.bias is not None:
                    expert.fc2.bias.copy_(dense_ffn.c_proj.bias)
