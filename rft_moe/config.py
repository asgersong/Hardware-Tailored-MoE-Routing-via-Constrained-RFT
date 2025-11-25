from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
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


@dataclass
class TrainingConfig:
    batch_size: int = 16
    block_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_iters: int = 1000
    log_interval: int = 20
    device: str = "cuda"
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
    ckpt_path: Optional[str] = None


@dataclass
class RFTConfig:
    ppo_epochs: int = 2
    ppo_clip: float = 0.1
    entropy_bonus: float = 1e-3
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    rollout_steps: int = 256
    rcpo_lambda: float = 0.1
    rcpo_lr: float = 5e-3
    quality_epsilon: float = 0.1
    latency_weight: float = 1.0
    quality_weight: float = 1.0
    actor_sync_interval: int = 50

