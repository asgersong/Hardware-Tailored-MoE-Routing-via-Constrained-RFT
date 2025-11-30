import argparse
import os
import sys

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Ensure project root is importable when running as a script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rft_moe.data.dataset import build_dataloader, load_tinystories_tokens
from rft_moe.modeling.gpt import GPT, GPTConfig
from rft_moe.utils.common import get_device
from rft_moe.virtual_cluster import VirtualClusterConfig, VirtualClusterEnv


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate average virtual latency for MoE router vs random routing.")
    p.add_argument("--model_ckpt", type=str, required=True, help="MoE checkpoint path")
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_batches", type=int, default=20)
    p.add_argument("--tokenizer", type=str, default="gpt2")
    p.add_argument("--num_experts", type=int, default=8)
    p.add_argument("--num_nodes", type=int, default=2)
    p.add_argument("--moe_top_k", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    tokens, vocab_size = load_tinystories_tokens(
        tokenizer_name=args.tokenizer, block_size=args.block_size, num_samples=50000
    )
    dl = build_dataloader(
        tokens, block_size=args.block_size, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    cluster = VirtualClusterEnv(
        VirtualClusterConfig(num_experts=args.num_experts, num_nodes=args.num_nodes)
    )
    moe_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        use_moe=True,
        num_experts=args.num_experts,
        num_nodes=args.num_nodes,
        moe_top_k=args.moe_top_k,
    )
    model = GPT(moe_config).to(device)
    ckpt = torch.load(args.model_ckpt, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    lat_router, lat_random, count = 0.0, 0.0, 0
    with torch.no_grad():
        for idx, (x, _) in enumerate(dl):
            if idx >= args.num_batches:
                break
            x = x.to(device)
            token_nodes = torch.randint(
                low=0, high=cluster.config.num_nodes, size=(x.size(0), x.size(1)), device=device
            )
            _, _, routers = model(x, targets=None, token_nodes=token_nodes, return_router_outputs=True)
            if not routers:
                continue
            router = routers[0]
            selected = router.selected_experts  # (B, T, k)
            B, T, K = selected.shape

            # Router latency
            lat = cluster.compute_latency(
                token_nodes=token_nodes.reshape(-1),
                selected_experts=selected.reshape(-1, K),
            ).mean()
            lat_router += lat.item() * B * T

            # Random baseline latency
            rand_experts = torch.randint(
                low=0, high=cluster.config.num_experts, size=(B, T, K), device=device
            )
            lat_r = cluster.compute_latency(
                token_nodes=token_nodes.reshape(-1),
                selected_experts=rand_experts.reshape(-1, K),
            ).mean()
            lat_random += lat_r.item() * B * T

            count += B * T

    lat_router = lat_router / max(count, 1)
    lat_random = lat_random / max(count, 1)
    print(f"Avg virtual latency (router): {lat_router:.4f}")
    print(f"Avg virtual latency (random): {lat_random:.4f}")
    if lat_random > 0:
        improvement = (lat_random - lat_router) / lat_random * 100.0
        print(f"Latency reduction vs random: {improvement:.2f}%")


if __name__ == "__main__":
    main()
