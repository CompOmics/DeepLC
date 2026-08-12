"""
Composition-encoder multitask architecture.

An alternative to :class:`deeplc._architecture.MultitaskDeepLCModel`, differing
in three ways that were each measured:

``pointwise stem``
    A kernel-1 convolution decodes each position's six atom counts on its own,
    before any neighbour mixing. The main convolutions use kernel 5 and would
    otherwise have to disentangle residue identity and sequence context at the
    same time.
``low-rank read-out``
    Each LC setup is a 64-dimensional vector dotted with a projected trunk,
    rather than an independent head. A new setup costs 66 parameters, and a
    calibration fitted in that space converges to the trained values rather
    than approximating them.
``corrected global features``
    Requires ``matrix_global`` with the terminal-composition blocks, that is
    ``encode_peptidoform(..., add_terminal_composition=True)``. Hence
    :attr:`CompositionMultitaskModel.requires_terminal_composition`.

``matrix_sum`` is accepted for signature compatibility and ignored: it equals
``matrix.reshape(30, 2, 6).sum(1)`` exactly, so it carries nothing the atom
matrix does not already provide.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

PAD_INDEX = 20
N_ATOMS = 6


class InputNorm(nn.Module):
    """Standardise dense inputs with statistics fitted on the training set."""

    def __init__(self, n_features: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardise ``x`` with the stored statistics."""
        return (x - self.mean) / self.std


class ConvBlock(nn.Module):
    """One convolution stage: convolution then SiLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution and activation."""
        return self.act(self.conv(x))


class FactorHead(nn.Module):
    """
    Low-rank read-out: a per-setup vector dotted with a projected trunk.

    ``prediction[:, j] = (proj(trunk) . embedding[j]) * scale[j] + shift[j]``
    """

    def __init__(self, trunk_dim: int, n_tasks: int, rank: int = 64):
        super().__init__()
        self.proj = nn.Linear(trunk_dim, rank)
        self.embedding = nn.Parameter(torch.zeros(n_tasks, rank))
        self.scale = nn.Parameter(torch.ones(n_tasks))
        self.shift = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, trunk: torch.Tensor) -> torch.Tensor:
        """Return one prediction per LC setup."""
        return self.proj(trunk) @ self.embedding.t() * self.scale + self.shift


class CompositionEncoder(nn.Module):
    """Peptide encoder: pointwise stem, convolution stack, masked pooling, dense trunk."""

    def __init__(
        self,
        global_dim: int = 67,
        embed_dim: int = 16,
        channels: Sequence[int] = (512, 512),
        kernel_size: int = 5,
        stem_pointwise: int = 128,
        stem_layers: int = 2,
        width: int = 256,
        depth: int = 3,
    ):
        super().__init__()
        self.embed = nn.Embedding(PAD_INDEX + 1, embed_dim, padding_idx=PAD_INDEX)
        self.stem = nn.Sequential(
            *[
                ConvBlock(N_ATOMS if i == 0 else stem_pointwise, stem_pointwise, 1)
                for i in range(stem_layers)
            ]
        )

        in_channels = stem_pointwise + embed_dim
        blocks = []
        for out_channels in channels:
            blocks.append(ConvBlock(in_channels, out_channels, kernel_size))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

        pooled_dim = in_channels * 2  # sum and max pooling
        self.pool_norm = nn.LayerNorm(pooled_dim)
        dense_dim = global_dim + PAD_INDEX  # global features and residue counts
        self.norm = InputNorm(dense_dim)

        layers: list[nn.Module] = []
        previous = dense_dim + pooled_dim
        for _ in range(depth):
            layers.extend([nn.Linear(previous, width), nn.SiLU()])
            previous = width
        self.net = nn.Sequential(*layers)
        self.trunk_dim = width

    def forward(
        self, x_atom: torch.Tensor, x_global: torch.Tensor, residue_index: torch.Tensor
    ) -> torch.Tensor:
        """Encode a batch of peptides into trunk vectors."""
        valid = (residue_index != PAD_INDEX).unsqueeze(1)

        hidden = torch.cat(
            [self.stem(x_atom.transpose(1, 2)), self.embed(residue_index).transpose(1, 2)], dim=1
        )
        for block in self.blocks:
            hidden = block(hidden)
        hidden = hidden * valid

        pooled = torch.cat(
            [
                hidden.sum(dim=2),
                torch.nan_to_num(
                    hidden.masked_fill(~valid, float("-inf")).max(dim=2).values, neginf=0.0
                ),
            ],
            dim=1,
        )
        pooled = self.pool_norm(pooled)

        counts = (
            nn.functional.one_hot(residue_index.clamp(0, PAD_INDEX), PAD_INDEX + 1)
            .sum(1)[:, :PAD_INDEX]
            .float()
        )
        dense = self.norm(torch.cat([x_global.float(), counts], dim=1))
        return self.net(torch.cat([dense, pooled], dim=1))


class CompositionMultitaskModel(nn.Module):
    """
    Composition encoder with a low-rank multitask read-out.

    Takes the same four feature tensors as :class:`DeepLCModel` so it is a drop-in
    for the prediction path, and returns ``(batch, n_tasks)`` predictions.
    """

    #: ``matrix_global`` must carry the terminal-composition blocks for this model.
    requires_terminal_composition = True

    def __init__(
        self, n_tasks: int = 1025, rank: int = 64, global_dim: int = 67, **encoder_kwargs
    ):
        super().__init__()
        self.encoder = CompositionEncoder(global_dim=global_dim, **encoder_kwargs)
        self.head = FactorHead(self.encoder.trunk_dim, n_tasks, rank=rank)

    def forward(
        self,
        x_atom: torch.Tensor,
        x_atom_sum: torch.Tensor,  # noqa: ARG002 - accepted for signature compatibility
        x_global: torch.Tensor,
        x_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        """Predict retention time for every LC setup the model was trained on."""
        residue_index = x_one_hot.argmax(dim=2).long()
        residue_index = residue_index.masked_fill(x_one_hot.sum(dim=2) == 0, PAD_INDEX)
        return self.head(self.encoder(x_atom, x_global, residue_index))

    @classmethod
    def from_state_dict(cls, state: dict) -> CompositionMultitaskModel:
        """Rebuild the architecture from the shapes stored in a state dict."""
        channels = [
            state[key].shape[0]
            for key in state
            if key.startswith("encoder.blocks.") and key.endswith(".conv.weight")
        ]
        stem_layers = sum(
            1 for key in state if key.startswith("encoder.stem.") and key.endswith(".conv.weight")
        )
        model = cls(
            n_tasks=state["head.embedding"].shape[0],
            rank=state["head.embedding"].shape[1],
            global_dim=state["encoder.norm.mean"].shape[0] - PAD_INDEX,
            embed_dim=state["encoder.embed.weight"].shape[1],
            channels=channels,
            kernel_size=state["encoder.blocks.0.conv.weight"].shape[2],
            stem_pointwise=state["encoder.stem.0.conv.weight"].shape[0],
            stem_layers=stem_layers,
            width=state["encoder.net.0.weight"].shape[0],
            depth=sum(
                1 for key in state if key.startswith("encoder.net.") and key.endswith(".weight")
            ),
        )
        model.load_state_dict(state)
        return model
