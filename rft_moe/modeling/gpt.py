from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import MoELayer, RouterOutput


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    use_moe: bool = False
    moe_every: int = 2
    moe_top_k: int = 2
    num_experts: int = 8
    router_init_std: float = 0.02
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        )
        self.register_buffer("bias", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig, use_moe: bool = False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.use_moe = use_moe
        if use_moe:
            self.mlp = MoELayer(
                dim=config.n_embd,
                num_experts=config.num_experts,
                k=config.moe_top_k,
                router_init_std=config.router_init_std,
                dropout=config.dropout,
                bias=config.bias,
            )
        else:
            self.mlp = FeedForward(config)

    def forward(
        self, x: torch.Tensor, return_router_outputs: bool = False
    ) -> Tuple[torch.Tensor, List[RouterOutput]]:
        router_outputs: List[RouterOutput] = []
        h = self.ln_1(x)
        x = x + self.attn(h)
        h = self.ln_2(x)
        residual = x
        if self.use_moe:
            x, router_out = self.mlp(h, return_router_outputs=return_router_outputs)
            x = residual + x
            if router_out is not None:
                router_outputs.append(router_out)
        else:
            x = x + self.mlp(h)
        return x, router_outputs


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        for layer_idx in range(config.n_layer):
            use_moe = config.use_moe and (layer_idx % config.moe_every == 0)
            self.transformer["h"].append(Block(config, use_moe=use_moe))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_router_outputs: bool = False,
    ):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, "Sequence length exceeds block size"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer["wte"](idx)
        pos_emb = self.transformer["wpe"](pos)
        x = tok_emb + pos_emb
        x = self.transformer["drop"](x)

        all_router_outputs: List[RouterOutput] = []
        for block in self.transformer["h"]:
            x, router_outputs = block(x, return_router_outputs=return_router_outputs)
            if router_outputs:
                all_router_outputs.extend(router_outputs)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="mean"
            )
        return logits, loss, all_router_outputs

    def configure_optimizers(self, weight_decay: float, lr: float, betas=(0.9, 0.95)):
        decay, no_decay = [], []
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias"):
                    no_decay.append(p)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.append(p)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.append(p)
        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer

    def freeze_except_router(self):
        for name, param in self.named_parameters():
            if ("mlp.router" in name) or ("mlp.value_head" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
