"""Diagnostic fp32 pairwise reduction; not an accepted NeMo processor replacement.

Scripted control flow preserves dynamic frame lengths in the ONNX Loop.
Inputs are finite float32 [batch, bins, frames], with 1 <= valid <= frames.
The caller validates that boundary before export/inference.
"""
import torch


@torch.jit.script
def pairwise_sum(values: torch.Tensor) -> torch.Tensor:
    current = values
    while current.size(-1) > 1:
        if current.size(-1) % 2 != 0:
            current = torch.cat((current, torch.zeros_like(current[:, :, :1])), dim=-1)
        current = current[:, :, 0::2] + current[:, :, 1::2]
    return current.squeeze(-1)


class PairwiseNormalization(torch.nn.Module):
    def __init__(self, pad_value=0.0):
        super().__init__()
        self.pad_value = float(pad_value)

    def forward(self, log_mel, valid):
        mask = torch.arange(log_mel.size(-1), device=log_mel.device)[None, None, :] < valid[:, None, None]
        masked = torch.where(mask, log_mel, torch.zeros_like(log_mel))
        count = valid.to(log_mel.dtype)[:, None]
        mean = pairwise_sum(masked) / count
        delta = torch.where(mask, log_mel - mean[:, :, None], torch.zeros_like(log_mel))
        variance = pairwise_sum(delta * delta) / (count - 1)
        std = torch.sqrt(variance)
        std = torch.where(torch.isnan(std), torch.zeros_like(std), std) + 1e-5
        features = (log_mel - mean[:, :, None]) / std[:, :, None]
        return torch.where(mask, features, torch.full_like(features, self.pad_value)), mean, std
