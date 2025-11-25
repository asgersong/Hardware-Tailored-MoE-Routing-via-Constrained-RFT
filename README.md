# Accelerating LLMs with Hardware-Tailored MoE Routing via Constrained RFT

End-to-end code skeleton for the feasibility study "Hardware-Aware Mixture-of-Experts Routing via Constrained Reinforcement Fine-Tuning." The repo contains:
- Tiny GPT-style LM (dense baseline) and MoE variant with upcycled experts.
- Virtual cluster latency simulator for hardware-aware rewards.
- PPO/RCPO router fine-tuning loop and distributed actor-learner scaffold for 5x GPU throughput.
- Evaluation helper to inspect locality heatmaps.

## Repo layout
- `rft_moe/modeling`: GPT blocks, MoE layer, router outputs.
- `rft_moe/data`: TinyStories loader and dataloader helpers.
- `rft_moe/training/train_dense.py`: Dense TinyGPT baseline trainer.
- `rft_moe/training/train_rft.py`: Router PPO/RCPO fine-tuning over virtual latency.
- `rft_moe/training/upcycle.py`: Sparse upcycling utility (dense -> MoE).
- `rft_moe/training/distributed.py`: Actor-learner scaffolding for multi-GPU rollout collection.
- `rft_moe/virtual_cluster.py`: Virtual cluster cost model.
- `scripts/eval_routing.py`: Routing locality/heatmap inspection.
- `requirements.txt`: Minimal deps.

For multi-GPU throughput, wrap the rollout_fn/update_fn with `rft_moe.training.distributed.learner_loop` to run 4 actors (GPUs 1-4) and a learner (GPU0).

### Monitoring with MLflow
- Install MLflow (`pip install mlflow`) and enable logging with `--mlflow`. Optional args: `--mlflow_tracking_uri`, `--mlflow_experiment`, `--mlflow_run_name`.
- Example: `python -m rft_moe.training.train_dense --mlflow --mlflow_run_name dense-baseline`
- Logs: loss/perplexity (dense) or reward/quality/lambda (router RFT), plus checkpoint artifact for the dense run.

## Evaluation
Inspect routing locality/latency after RFT:
`python scripts/eval_routing.py --model_ckpt checkpoints/moe_upcycled.pt`

Output: normalized token-node -> expert-node heatmap to verify locality bias and reduced virtual latency.

## Notes
- TinyStories is loaded via Hugging Face (`datasets` + `transformers`). Pre-download once; then all runs are local.
- Default configs keep experts frozen and train only router/value heads. Adjust `GPTConfig` if you wish to unfreeze experts for ablations.
- Virtual cluster costs are in `rft_moe/virtual_cluster.py`; tweak inter/intra latency or load penalty to emulate different interconnects (PCIe vs NVLink).
