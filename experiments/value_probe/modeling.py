"""Torch modules for token-level value probes."""

from __future__ import annotations


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "This command requires torch. Install the probe dependencies first, "
            "for example: uv pip install torch transformers numpy tqdm safetensors accelerate"
        ) from exc
    return torch, nn


torch, nn = require_torch()


class TokenValueProbe(nn.Module):
    """A linear token probe with positive and opposite-evidence heads."""

    def __init__(self, hidden_size: int, num_values: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.positive_head = nn.Linear(hidden_size, num_values)
        self.negative_head = nn.Linear(hidden_size, num_values)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.dropout(self.norm(hidden_states.float()))
        return self.positive_head(hidden_states), self.negative_head(hidden_states)


class FluentValueProbe(nn.Module):
    """A linear token probe that regresses signed value relevance scores."""

    def __init__(self, hidden_size: int, num_values: int, dropout: float = 0.0, score_scale: float = 6.0) -> None:
        super().__init__()
        self.score_scale = float(score_scale)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.score_head = nn.Linear(hidden_size, num_values)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(self.norm(hidden_states.float()))
        return self.score_scale * torch.tanh(self.score_head(hidden_states))


def masked_logmeanexp(logits: torch.Tensor, attention_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    """Aggregate token logits into document logits while keeping token attribution."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    mask = attention_mask.bool().unsqueeze(-1)
    scaled = logits / temperature
    scaled = scaled.masked_fill(~mask, -1e4)
    token_counts = mask.sum(dim=1).clamp_min(1).to(logits.dtype)
    return (torch.logsumexp(scaled, dim=1) - token_counts.log()) * temperature


def masked_topk_mean(logits: torch.Tensor, attention_mask: torch.Tensor, k: int) -> torch.Tensor:
    """Aggregate token logits with the mean of the top-k valid token scores."""

    if k <= 0:
        raise ValueError("k must be positive")
    mask = attention_mask.bool().unsqueeze(-1)
    masked = logits.masked_fill(~mask, -1e4)
    topk = min(k, logits.shape[1])
    values, _ = masked.topk(topk, dim=1)
    valid = values > -9999
    denom = valid.sum(dim=1).clamp_min(1)
    return (values.masked_fill(~valid, 0.0).sum(dim=1) / denom).to(logits.dtype)


def aggregate(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    method: str,
    temperature: float,
    topk: int,
) -> torch.Tensor:
    if method == "logmeanexp":
        return masked_logmeanexp(logits, attention_mask, temperature)
    if method == "topk_mean":
        return masked_topk_mean(logits, attention_mask, topk)
    raise ValueError(f"Unsupported aggregation method: {method}")
