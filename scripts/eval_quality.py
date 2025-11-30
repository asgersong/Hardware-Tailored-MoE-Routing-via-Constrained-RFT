import argparse
import os
import sys

import torch
import torch.nn.functional as F

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rft_moe.data.dataset import build_dataloader, load_tinystories_tokens
from rft_moe.modeling.gpt import GPT, GPTConfig
from rft_moe.utils.common import get_device


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CE/PPL and teacher KL on TinyStories.")
    p.add_argument("--dense_ckpt", type=str, required=True, help="Path to dense baseline checkpoint.")
    p.add_argument("--moe_ckpt", type=str, required=True, help="Path to MoE/RFT checkpoint.")
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_batches", type=int, default=50, help="How many batches to evaluate.")
    p.add_argument("--tokenizer", type=str, default="gpt2")
    p.add_argument("--num_experts", type=int, default=8)
    p.add_argument("--num_nodes", type=int, default=2)
    p.add_argument("--num_samples", type=int, default=50000, help="Limit TinyStories samples for faster eval.")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print(
        f"Loading TinyStories (num_samples={args.num_samples}, batch_size={args.batch_size})..."
    )
    tokens, vocab_size = load_tinystories_tokens(
        tokenizer_name=args.tokenizer, block_size=args.block_size, num_samples=args.num_samples
    )
    dl = build_dataloader(
        tokens, block_size=args.block_size, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    dense_cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        use_moe=False,
    )
    moe_cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        use_moe=True,
        num_experts=args.num_experts,
        num_nodes=args.num_nodes,
    )

    dense = GPT(dense_cfg).to(device)
    dense_ckpt = torch.load(args.dense_ckpt, map_location=device)
    dense.load_state_dict(dense_ckpt.get("model", dense_ckpt))
    dense.eval()

    moe = GPT(moe_cfg).to(device)
    moe_ckpt = torch.load(args.moe_ckpt, map_location=device)
    moe.load_state_dict(moe_ckpt.get("model", moe_ckpt))
    moe.eval()

    ce_dense, ce_moe, kl = [], [], []
    with torch.no_grad():
        for idx, (x, y) in enumerate(dl):
            if idx >= args.num_batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits_dense, _, _ = dense(x, targets=None, return_router_outputs=False)
            logits_moe, _, _ = moe(x, targets=None, return_router_outputs=False)

            # CE
            eff_len = min(logits_dense.size(1) - 1, y.size(1))
            ce_d = F.cross_entropy(
                logits_dense[:, :eff_len, :].reshape(-1, logits_dense.size(-1)),
                y[:, :eff_len].reshape(-1),
                reduction="mean",
            )
            ce_m = F.cross_entropy(
                logits_moe[:, :eff_len, :].reshape(-1, logits_moe.size(-1)),
                y[:, :eff_len].reshape(-1),
                reduction="mean",
            )
            ce_dense.append(ce_d.item())
            ce_moe.append(ce_m.item())

            # Teacher KL: KL(dense || moe) over token positions
            p = torch.log_softmax(logits_dense[:, :eff_len, :], dim=-1)
            q = torch.log_softmax(logits_moe[:, :eff_len, :], dim=-1)
            kl_tok = torch.exp(p) * (p - q)
            kl_val = kl_tok.sum(dim=-1).mean()
            kl.append(kl_val.item())

    mean_ce_dense = sum(ce_dense) / len(ce_dense)
    mean_ce_moe = sum(ce_moe) / len(ce_moe)
    mean_kl = sum(kl) / len(kl)
    ppl_dense = float(torch.exp(torch.tensor(mean_ce_dense)))
    ppl_moe = float(torch.exp(torch.tensor(mean_ce_moe)))
    ce_delta_pct = (mean_ce_moe - mean_ce_dense) / mean_ce_dense * 100.0
    ppl_delta_pct = (ppl_moe - ppl_dense) / ppl_dense * 100.0

    print(f"Eval over {len(ce_dense)} batches of size {args.batch_size}")
    print(f"Dense CE: {mean_ce_dense:.4f} | PPL: {ppl_dense:.2f}")
    print(f"MoE   CE: {mean_ce_moe:.4f} | PPL: {ppl_moe:.2f}")
    print(f"Delta CE: {ce_delta_pct:+.2f}% | Delta PPL: {ppl_delta_pct:+.2f}% vs dense")
    print(f"KL(dense || moe): {mean_kl:.4f}")


if __name__ == "__main__":
    main()
