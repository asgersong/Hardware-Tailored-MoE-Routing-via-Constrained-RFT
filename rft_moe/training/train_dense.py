import argparse
from pathlib import Path
import os
import warnings

import torch
from torch.cuda.amp import GradScaler, autocast

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rft_moe.config import ModelConfig, TrainingConfig
from rft_moe.data.dataset import build_dataloader, load_tinystories_tokens
from rft_moe.modeling.gpt import GPT, GPTConfig
from rft_moe.utils.common import get_device, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train dense TinyGPT baseline.")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_iters", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="checkpoints/dense_baseline.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging if installed.")
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None)
    parser.add_argument("--mlflow_experiment", type=str, default="rft-moe")
    parser.add_argument("--mlflow_run_name", type=str, default="dense-baseline")
    return parser.parse_args()


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
                    "block_size": args.block_size,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "max_iters": args.max_iters,
                    "tokenizer": args.tokenizer,
                    "num_samples": args.num_samples,
                }
            )

    tokens, vocab_size = load_tinystories_tokens(
        tokenizer_name=args.tokenizer, block_size=args.block_size, num_samples=args.num_samples
    )
    dl = build_dataloader(tokens, block_size=args.block_size, batch_size=args.batch_size)

    model_cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        use_moe=False,
    )
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        lr=args.lr,
        max_iters=args.max_iters,
        log_interval=args.log_interval,
    )

    model = GPT(model_cfg).to(device)
    optimizer = model.configure_optimizers(weight_decay=0.1, lr=train_cfg.lr)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    it = iter(dl)
    for step in range(train_cfg.max_iters):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=torch.cuda.is_available()):
            logits, loss, _ = model(x, targets=y, return_router_outputs=False)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % train_cfg.log_interval == 0:
            ppl = torch.exp(loss.detach()).item()
            print(f"step {step} | loss {loss.item():.4f} | ppl {ppl:.2f}")
            if mlflow_run:
                mlflow.log_metrics({"loss": loss.item(), "ppl": ppl}, step=step)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, args.save_path)
    print(f"Saved dense baseline to {args.save_path}")
    if mlflow_run:
        mlflow.log_artifact(args.save_path)
        mlflow.end_run()


if __name__ == "__main__":
    main()
