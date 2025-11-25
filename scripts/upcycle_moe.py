import argparse

from torch.optim import AdamW

from rft_moe.modeling.gpt import GPTConfig
from rft_moe.training.upcycle import build_moe_from_dense
from rft_moe.utils.common import get_device, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Upcycle dense checkpoint into MoE checkpoint.")
    parser.add_argument("--dense_ckpt", type=str, required=True, help="Path to dense baseline checkpoint.")
    parser.add_argument("--save_path", type=str, default="checkpoints/moe_upcycled.pt", help="Where to save MoE checkpoint.")
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--moe_top_k", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4, help="LR for the saved optimizer state.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    cfg = GPTConfig(
        use_moe=True,
        num_experts=args.num_experts,
        moe_top_k=args.moe_top_k,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    moe = build_moe_from_dense(args.dense_ckpt, cfg, device=device)
    # Optimizer only over trainable params (router/value heads by default after freeze_except_router in training script).
    opt = AdamW(filter(lambda p: p.requires_grad, moe.parameters()), lr=args.lr)

    save_checkpoint(moe, opt, args.save_path)
    print(f"MoE upcycled checkpoint saved to {args.save_path}")


if __name__ == "__main__":
    main()
