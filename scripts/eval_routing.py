import argparse
import os
import sys

import numpy as np
import torch

# Avoid tokenizers fork warnings and ensure project root is importable.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rft_moe.data.dataset import build_dataloader, load_tinystories_tokens
from rft_moe.modeling.gpt import GPT, GPTConfig
from rft_moe.utils.common import get_device
from rft_moe.virtual_cluster import VirtualClusterConfig, VirtualClusterEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate routing locality and latency.")
    parser.add_argument("--model_ckpt", type=str, required=True, help="MoE checkpoint path")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--num_nodes", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    tokens, vocab_size = load_tinystories_tokens(tokenizer_name=args.tokenizer, block_size=args.block_size, num_samples=2000)
    dl = build_dataloader(tokens, block_size=args.block_size, batch_size=args.batch_size, shuffle=False)

    cluster = VirtualClusterEnv(
        VirtualClusterConfig(num_experts=args.num_experts, num_nodes=args.num_nodes)
    )
    moe_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        use_moe=True,
        num_experts=args.num_experts,
        num_nodes=cluster.config.num_nodes,
    )
    model = GPT(moe_config).to(device)
    ckpt = torch.load(args.model_ckpt, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    heatmap = np.zeros((cluster.config.num_nodes, cluster.config.num_nodes), dtype=np.float64)
    total = 0
    with torch.no_grad():
        for idx, (x, _) in enumerate(dl):
            if idx >= args.num_batches:
                break
            x = x.to(device)
            token_nodes = torch.randint(
                low=0, high=cluster.config.num_nodes, size=(x.size(0), x.size(1)), device=device
            )
            logits, _, routers = model(
                x, targets=None, token_nodes=token_nodes, return_router_outputs=True
            )
            if not routers:
                continue
            router = routers[0]
            seq = x.size(1)
            selected = router.selected_experts  # (B, T, k)
            expert_nodes = cluster.expert_to_node.to(device)[selected]  # (B, T, k)
            for src in range(cluster.config.num_nodes):
                for dst in range(cluster.config.num_nodes):
                    count = (
                        (token_nodes.unsqueeze(-1) == src) & (expert_nodes == dst)
                    ).sum().item()
                    heatmap[src, dst] += count
            total += x.numel() * selected.size(-1)

    heatmap = heatmap / max(total, 1)
    print("Routing heatmap (token node -> expert node) normalized by token-slots:")
    for row in heatmap:
        print(" ".join(f"{v:.4f}" for v in row))


if __name__ == "__main__":
    main()
