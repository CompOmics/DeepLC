"""Utilities for adapting multitask RT models to single-task outputs."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class MultitaskAdapter(nn.Module):
    """Wrap a multitask backbone and map its head vector to one RT output."""

    def __init__(self, multitask_model: nn.Module, n_heads: int, hidden_size: int = 256):
        super().__init__()
        self.backbone = copy.deepcopy(multitask_model)
        self.adapter = nn.Sequential(
            nn.Linear(n_heads, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, max(1, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(1, hidden_size // 2), 1),
        )

    def forward(self, x_atom, x_atom_sum, x_global, x_one_hot):
        multitask_output = self.backbone(x_atom, x_atom_sum, x_global, x_one_hot)
        if multitask_output.ndim == 1:
            multitask_output = multitask_output.unsqueeze(-1)
        return self.adapter(multitask_output)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
