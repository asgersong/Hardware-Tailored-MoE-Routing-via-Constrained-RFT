import argparse
import os
import warnings
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rft_moe.config import RFTConfig
from rft_moe.data.dataset import build_dataloader, load_tinystories_tokens
from rft_moe.modeling.gpt import GPTConfig
from rft_moe.training.ppo import (
    compute_advantages,
    entropy_from_logits,
    ppo_objective,
    update_rcpo_lambda,
    value_loss,
)
from rft_moe.training.upcycle import build_moe_from_dense
from rft_moe.utils.common import get_device, save_checkpoint, set_seed
from rft_moe.virtual_cluster import VirtualClusterConfig, VirtualClusterEnv


@dataclass
class RolloutSample:
    input_ids: torch.Tensor
    targets: torch.Tensor
    token_nodes: torch.Tensor
    selected_experts: torch.Tensor
    old_logp: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    quality: torch.Tensor
    latency: torch.Tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Router RFT with PPO/RCPO.")
    parser.add_argument("--dense_ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rollout_steps", type=int, default=64)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--max_updates", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--moe_top_k", type=int, default=2)
    parser.add_argument("--lambda_quality", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save RFT checkpoint.")
    parser.add_argument("--fixed_lambda", action="store_true", help="If set, keep lambda constant at lambda_quality (disable RCPO updates).")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging if installed.")
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None)
    parser.add_argument("--mlflow_experiment", type=str, default="rft-moe")
    parser.add_argument("--mlflow_run_name", type=str, default="router-rft")
    return parser.parse_args()


def gather_log_prob_from_logits(logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    gathered = []
    for slot in range(selected_experts.size(-1)):
        idx = selected_experts[..., slot]
        slot_prob = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        gathered.append(torch.log(slot_prob + 1e-9))
    return torch.stack(gathered, dim=-1).sum(dim=-1)


def collect_rollouts(
    model,
    dataloader,
    cluster: VirtualClusterEnv,
    rft_cfg: RFTConfig,
    device: torch.device,
    lambda_quality: torch.Tensor,
) -> List[RolloutSample]:
    rollouts: List[RolloutSample] = []
    data_iter = iter(dataloader)
    for _ in range(rft_cfg.rollout_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        x = x.to(device)
        y = y.to(device)

        model.eval()
        with torch.no_grad():
            token_nodes = torch.randint(
                low=0,
                high=cluster.config.num_nodes,
                size=(x.size(0), x.size(1)),
                device=device,
            )
            logits, _, router_outputs = model(
                x, targets=None, token_nodes=token_nodes, return_router_outputs=True
            )
        if not router_outputs:
            raise RuntimeError("Model returned no router outputs; ensure use_moe=True and moe_every>0.")
        router_out = router_outputs[0]

        seq = x.size(1)
        tgt_len = y.size(1)
        eff_len = min(seq, tgt_len)
        ce = F.cross_entropy(
            logits[:, :eff_len, :].reshape(-1, logits.size(-1)),
            y[:, :eff_len].reshape(-1),
            reduction="none",
        ).view(x.size(0), eff_len)
        selected = router_out.selected_experts[:, :eff_len, :]
        latency = cluster.compute_latency(
            token_nodes=token_nodes.reshape(-1),
            selected_experts=selected.reshape(-1, selected.size(-1)),
        ).view(x.size(0), eff_len)
        rewards = -rft_cfg.latency_weight * latency - lambda_quality * ce
        rollouts.append(
            RolloutSample(
                input_ids=x.detach(),
                targets=y.detach(),
                token_nodes=token_nodes.detach(),
                selected_experts=selected.detach(),
                old_logp=router_out.log_prob[:, :eff_len].detach(),
                rewards=rewards.detach(),
                values=router_out.value[:, :eff_len].detach(),
                quality=ce.detach(),
                latency=latency.detach(),
            )
        )
    return rollouts


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    mlflow_run = None
    if args.mlflow:
        try:
            import mlflow
            warnings.filterwarnings(
                "ignore",
                message="Filesystem tracking backend",
                category=FutureWarning,
            )
        except ImportError:
            print("MLflow is not installed; run `pip install mlflow` or disable --mlflow.")
            mlflow = None
        if mlflow:
            if args.mlflow_tracking_uri:
                mlflow.set_tracking_uri(args.mlflow_tracking_uri)
            mlflow.set_experiment(args.mlflow_experiment)
            mlflow_run = mlflow.start_run(run_name=args.mlflow_run_name)
            mlflow.log_params(
                {
                    "batch_size": args.batch_size,
                    "block_size": args.block_size,
                    "lr": args.lr,
                    "rollout_steps": args.rollout_steps,
                    "ppo_epochs": args.ppo_epochs,
                    "max_updates": args.max_updates,
                    "num_experts": args.num_experts,
                    "moe_top_k": args.moe_top_k,
                    "lambda_quality": args.lambda_quality,
                    "tokenizer": args.tokenizer,
                    "num_samples": args.num_samples,
                }
            )

    tokens, vocab_size = load_tinystories_tokens(
        tokenizer_name=args.tokenizer,
        block_size=args.block_size,
        num_samples=args.num_samples,
    )
    dataloader = build_dataloader(tokens, block_size=args.block_size, batch_size=args.batch_size)

    cluster = VirtualClusterEnv(VirtualClusterConfig(num_experts=args.num_experts))
    moe_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        use_moe=True,
        num_experts=args.num_experts,
        num_nodes=cluster.config.num_nodes,
        moe_top_k=args.moe_top_k,
    )
    model = build_moe_from_dense(args.dense_ckpt, moe_config, device=device)
    model.freeze_except_router()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    rft_cfg = RFTConfig(
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        rcpo_lambda=args.lambda_quality,
    )
    lambda_quality = torch.tensor(rft_cfg.rcpo_lambda, device=device)

    for update_idx in range(args.max_updates):
        rollouts = collect_rollouts(model, dataloader, cluster, rft_cfg, device, lambda_quality)

        model.train()
        for _ in range(rft_cfg.ppo_epochs):
            for batch in rollouts:
                logits, _, router_outputs = model(
                    batch.input_ids.to(device),
                    targets=None,
                    token_nodes=batch.token_nodes.to(device),
                    return_router_outputs=True,
                )
                if not router_outputs:
                    raise RuntimeError("Model returned no router outputs during training.")
                router_out = router_outputs[0]
                eff_len = min(router_out.logits.size(1), batch.targets.size(1))
                new_logp = gather_log_prob_from_logits(
                    router_out.logits[:, :eff_len, :], batch.selected_experts[:, :eff_len, :].to(device)
                )
                entropy = entropy_from_logits(router_out.logits)
                values = router_out.value[:, :eff_len]

                rewards = batch.rewards[:, :eff_len].to(device)
                advantages, returns = compute_advantages(rewards, values)

                policy_loss = ppo_objective(new_logp, batch.old_logp[:, :eff_len].to(device), advantages, rft_cfg.ppo_clip)
                v_loss = value_loss(values, returns)
                loss = policy_loss + rft_cfg.value_coef * v_loss - rft_cfg.entropy_bonus * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), rft_cfg.max_grad_norm)
                optimizer.step()

        avg_reward = torch.cat([b.rewards for b in rollouts], dim=0).mean().item()
        avg_quality = torch.cat([b.quality for b in rollouts], dim=0).mean().item()
        avg_latency = torch.cat([b.latency for b in rollouts], dim=0).mean().item()
        if not args.fixed_lambda:
            lambda_quality = update_rcpo_lambda(lambda_quality, torch.tensor(avg_quality, device=device), rft_cfg.quality_epsilon, rft_cfg.rcpo_lr)

        if update_idx % args.log_interval == 0:
            print(
                f"update {update_idx} | reward {avg_reward:.4f} | quality {avg_quality:.4f} | latency {avg_latency:.4f} | lambda {lambda_quality.item():.4f}"
            )
            if mlflow_run:
                mlflow.log_metrics(
                    {
                        "reward": avg_reward,
                        "quality_ce": avg_quality,
                        "latency": avg_latency,
                        "lambda_quality": lambda_quality.item(),
                    },
                    step=update_idx,
                )

    print("RFT training complete.")
    if mlflow_run:
        mlflow.end_run()
    if args.save_path:
        save_checkpoint(model, optimizer, args.save_path)
        if mlflow_run:
            # Best-effort artifact logging
            mlflow.log_artifact(args.save_path)
        print(f"Saved RFT checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
