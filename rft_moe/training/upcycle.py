from pathlib import Path
from typing import Optional

import torch

from rft_moe.modeling.gpt import GPT, GPTConfig
from rft_moe.utils.common import load_checkpoint


def build_moe_from_dense(
    dense_ckpt: str,
    moe_config: GPTConfig,
    device: Optional[torch.device] = None,
) -> GPT:
    device = device or torch.device("cpu")
    dense_cfg = GPTConfig(
        vocab_size=moe_config.vocab_size,
        block_size=moe_config.block_size,
        n_layer=moe_config.n_layer,
        n_head=moe_config.n_head,
        n_embd=moe_config.n_embd,
        dropout=moe_config.dropout,
        bias=moe_config.bias,
        use_moe=False,
    )
    dense_model = GPT(dense_cfg)
    ckpt = torch.load(dense_ckpt, map_location="cpu")
    dense_model.load_state_dict(ckpt.get("model", ckpt))

    moe_model = GPT(moe_config)

    with torch.no_grad():
        moe_model.transformer["wte"].weight.copy_(dense_model.transformer["wte"].weight)
        moe_model.transformer["wpe"].weight.copy_(dense_model.transformer["wpe"].weight)
        moe_model.transformer["ln_f"].load_state_dict(dense_model.transformer["ln_f"].state_dict())
        moe_model.lm_head.weight.copy_(dense_model.lm_head.weight)

        for dense_block, moe_block in zip(dense_model.transformer["h"], moe_model.transformer["h"]):
            moe_block.ln_1.load_state_dict(dense_block.ln_1.state_dict())
            moe_block.ln_2.load_state_dict(dense_block.ln_2.state_dict())
            moe_block.attn.load_state_dict(dense_block.attn.state_dict())
            if moe_block.use_moe:
                moe_block.mlp.load_from_dense(dense_block.mlp)
            else:
                moe_block.mlp.load_state_dict(dense_block.mlp.state_dict())

    moe_model.to(device)
    return moe_model

