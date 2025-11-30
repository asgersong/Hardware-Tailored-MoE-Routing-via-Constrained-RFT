# Hardware-Tailored MoE Routing via Constrained RFT (Feasibility Study)

This repo is a proof-of-concept showing that a Mixture-of-Experts router can be fine-tuned to favor hardware-local experts (virtual cluster) with minimal quality loss.

What’s here
- Dense TinyStories GPT baseline and an MoE variant with experts upcycled from the dense FFN.
- Virtual cluster latency simulator (intra vs. inter-node cost, configurable via env vars).
- PPO/RCPO router fine-tuning (router/value/node embeddings train; experts frozen by default).
- Evaluation scripts for routing locality (heatmap), quality (CE/PPL/KL), and virtual latency vs. random.
- Diagram: `diagrams/upcycling.png`; Report: `REPORT.md`.

Notes
- Set env vars (`VIRTUAL_CLUSTER_INTER_NODE`, `VIRTUAL_CLUSTER_LOAD_PENALTY`, etc.) to sweep latency/load trade-offs.
- MLflow logging with `--mlflow`; training prints reward/quality/latency/λ.
- Default keeps experts frozen; unfreeze in `GPTConfig` if you want expert specialization experiments.

Repo layout
- `rft_moe/modeling`: GPT/MoE layers, router/value.
- `rft_moe/data`: TinyStories loader and dataloader helpers.
- `rft_moe/training`: dense trainer, upcycling, PPO/RCPO RFT, distributed scaffold.
- `scripts`: evals (routing, quality, latency), upcycle helper.
- `diagrams`: upcycling and PPL/latency plots.
- `REPORT.md`: summary of pipeline, runs, and trade-offs.
