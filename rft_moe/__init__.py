"""
Hardware-Tailored MoE Routing via Constrained RFT.

This package provides:
- A tiny GPT-style language model with optional MoE feed-forward blocks.
- Utilities for sparse upcycling from a dense checkpoint.
- A virtual cluster latency model for hardware-aware rewards.
- PPO/RCPO training harnesses for router reinforcement learning.
"""

__all__ = ["config", "virtual_cluster", "modeling", "training", "utils"]
