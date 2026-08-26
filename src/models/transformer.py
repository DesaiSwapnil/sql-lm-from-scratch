"""
Decoder-only transformer: token embeddings → N × TransformerBlock → RMSNorm → lm_head.

Architecture specifics:
    - No learned position embeddings — position is encoded via RoPE inside each block's MHA.
    - Weight-tied lm_head: the output projection reuses the token embedding matrix.
      This keeps the ~100k-token vocabulary from consuming a disproportionate share of
      the 247M parameter budget (the embedding table is counted once, not twice).
    - Pre-norm throughout (RMSNorm before each sublayer).
    - Optional activation (gradient) checkpointing: set model.gradient_checkpointing = True
      before training to trade compute for VRAM — useful on V100/L4 that OOM otherwise.

Weight init:
    Linear weights: N(0, 0.02).
    Residual output projections (attn out_proj, mlp down): scaled by 1/sqrt(2 * n_blocks)
    following the GPT-2 paper to prevent gradient explosion at depth during early training.
    Embedding: N(0, 0.02) (shared with lm_head; only initialized once).

Parameter count at default config:
    n_embed=768, n_head=12, n_blocks=24, swiglu_hidden=2048, vocab_size=100352
    token_embed: 100352 × 768 = 77,070,336   (weight-tied, lm_head adds 0)
    per block:   4 × 768² (qkv+out) + 3 × 768 × 2048 (gate+up+down) + 2×768 (norms) = 7,079,424
    24 blocks:   169,906,176
    final norm:  768
    total:       ~246,977,280  ≈ 247M
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from src.models.transformer_block import TransformerBlock
from src.models.rms_norm import RMSNorm


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_head: int,
        n_blocks: int,
        context_length: int,
        swiglu_hidden: int,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.gradient_checkpointing = False

        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, n_head, context_length, swiglu_hidden)
            for _ in range(n_blocks)
        ])
        self.norm = RMSNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        # Weight tying: lm_head reuses the token embedding matrix.
        self.lm_head.weight = self.token_embed.weight

        self._init_weights(n_blocks)

    def _init_weights(self, n_blocks: int) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale residual output projections down to prevent gradient explosion at depth.
        # Factor 2 accounts for both residual branches (attn + mlp) per block.
        residual_scale = 0.02 / (2 * n_blocks) ** 0.5
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_scale)

    def forward_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """Run the backbone; return final hidden states after the last RMSNorm."""
        x = self.token_embed(idx)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx:       input token ids (B, T).
            targets:   next-token targets (B, T); same shape as idx, shifted by 1 by the caller.
            loss_mask: float/bool (B, T); loss averaged only over positions where mask is 1.
                       Used in SFT to train only on assistant tokens. None = train on all.
        Returns:
            (logits (B, T, vocab), loss scalar or None).
        """
        x = self.forward_hidden(idx)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            flat_logits  = logits.reshape(B * T, V)
            flat_targets = targets.reshape(B * T).long()
            if loss_mask is not None:
                flat_mask = loss_mask.reshape(B * T).bool()
                loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
            else:
                loss = F.cross_entropy(flat_logits, flat_targets)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """
        Autoregressively generate max_new_tokens tokens appended to idx.

        Simple implementation without KV caching: re-runs the full context each step.
        Correct but slow for long generations; a KV cache can be added later.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature       # (B, vocab)
            if top_p is not None and top_p < 1.0:
                logits = _top_p_filter(logits, top_p)
            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling: zero out logits outside the top-p probability mass."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens that push cumulative probability beyond top_p.
    # Shift by 1 so the token that *reaches* top_p is kept.
    remove = (cumprobs - F.softmax(sorted_logits, dim=-1)) > top_p
    sorted_logits[remove] = float("-inf")
    return torch.scatter(logits, -1, sorted_idx, sorted_logits)


if __name__ == "__main__":
    # Smoke test: tiny config on CPU, should run in a few seconds.
    print("=== Smoke test (tiny config, CPU) ===")
    model = Transformer(
        vocab_size=512, n_embed=128, n_head=4, n_blocks=2,
        context_length=64, swiglu_hidden=341,
    )
    n_smoke = sum(p.numel() for p in model.parameters())
    print(f"Smoke model params: {n_smoke:,}")

    x = torch.randint(0, 512, (2, 48))
    y = torch.randint(0, 512, (2, 48))
    logits, loss = model(x, y)
    assert logits.shape == (2, 48, 512), f"unexpected logits shape: {logits.shape}"
    assert loss is not None
    print(f"Forward OK | logits {logits.shape} | loss {loss.item():.4f}")

    # SFT masked loss
    mask = torch.zeros(2, 48)
    mask[:, 24:] = 1.0
    _, loss_sft = model(x, y, loss_mask=mask)
    print(f"Masked SFT loss OK | loss {loss_sft.item():.4f}")

    # Generation
    seed = torch.randint(0, 512, (1, 5))
    out = model.generate(seed, max_new_tokens=10, temperature=0.8, top_p=0.9)
    assert out.shape == (1, 15), f"unexpected generate shape: {out.shape}"
    print(f"Generate OK | output shape {out.shape}")

    # Gradient checkpointing path
    model.gradient_checkpointing = True
    _, loss_gc = model(x, y)
    loss_gc.backward()
    print(f"Grad checkpointing OK | loss {loss_gc.item():.4f}")
    model.gradient_checkpointing = False

    # Target model param count (no forward pass — just count).
    print("\n=== Target model (247M config) ===")
    big = Transformer(
        vocab_size=100352, n_embed=768, n_head=12, n_blocks=24,
        context_length=1024, swiglu_hidden=2048,
    )
    n = sum(p.numel() for p in big.parameters())
    print(f"Total parameters: {n:,}  (~{n / 1e6:.1f}M)")

    # Verify weight tying: lm_head and token_embed share the same storage.
    assert big.lm_head.weight.data_ptr() == big.token_embed.weight.data_ptr(), \
        "lm_head and token_embed are NOT weight-tied!"
    print("Weight tying verified.")

    # Component breakdown.
    embed_p  = big.token_embed.weight.numel()
    block_p  = sum(p.numel() for p in big.blocks[0].parameters())
    norm_p   = big.norm.weight.numel()
    # lm_head shares embed weight, so total = embed + n_blocks*block + norm
    computed = embed_p + 24 * block_p + norm_p
    print(f"  token_embed: {embed_p:,}")
    print(f"  per block:   {block_p:,}")
    print(f"  24 blocks:   {24 * block_p:,}")
    print(f"  final norm:  {norm_p:,}")
    print(f"  computed:    {computed:,}  (matches nn.Module total: {n == computed})")
    print("\nAll checks passed.")
