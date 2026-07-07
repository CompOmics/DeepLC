"""
PyTorch architecture definitions for DeepLC.

This module contains the neural network architectures used by DeepLC for
predicting peptide retention times based on atomic composition and other features.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LeakyReLUSaturation(nn.Module):
    """
    Leaky ReLU activation with saturation (max value clipping).

    This custom activation function applies leaky ReLU followed by clamping
    to a maximum value, matching the original TensorFlow implementation's behavior.

    Parameters
    ----------
    negative_slope
        Negative slope coefficient for leaky ReLU (default: 0.1)
    max_value
        Maximum output value for saturation (default: 20.0)

    """

    def __init__(self, negative_slope: float = 0.1, max_value: float = 20.0):
        super().__init__()
        self.negative_slope = negative_slope
        self.max_value = max_value
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky_relu(x)
        return torch.clamp(x, max=self.max_value)


class ConvBlock(nn.Module):
    """
    Convolutional block with two Conv1D layers and optional max pooling.

    Parameters
    ----------
    in_channels
        Number of input channels
    out_channels
        Number of output channels
    kernel_size
        Size of the convolutional kernel
    use_pooling
        Whether to apply max pooling after convolutions
    pool_size
        Size of the max pooling window (default: 2)
    regularizer_val
        L1 regularization coefficient (default: 0.000005)

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_pooling: bool = True,
        pool_size: int = 2,
        regularizer_val: float = 0.000005,
    ):
        super().__init__()
        self.use_pooling = use_pooling
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.activation1 = LeakyReLUSaturation()

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.activation2 = LeakyReLUSaturation()

        self.pool: nn.MaxPool1d | None = None
        if use_pooling:
            self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

        # Store regularizer value for potential use in training
        self.regularizer_val = regularizer_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class GlobalFeatureBranch(nn.Module):
    """
    Dense network branch for processing global peptide features.

    Parameters
    ----------
    input_size
        Size of the input feature vector
    num_layers
        Number of dense layers
    hidden_size
        Layer size for each hidden layer
    regularizer_val
        L1 regularization coefficient (default: 0.000005)

    """

    def __init__(
        self,
        input_size: int,
        num_layers: int = 4,
        hidden_size: int = 64,
        regularizer_val: float = 0.000005,
    ):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(input_size if len(layers) == 0 else hidden_size, hidden_size),
                    LeakyReLUSaturation(),
                ]
            )

        self.network = nn.Sequential(*layers)
        self.regularizer_val = regularizer_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class OneHotBranch(nn.Module):
    """
    Convolutional branch for processing one-hot encoded amino acid sequences.

    This branch uses tanh activation instead of leaky ReLU and processes
    one-hot encoded amino acid features.

    Parameters
    ----------
    input_channels
        Number of input channels (20 for standard amino acids)
    sequence_length
        Length of the input sequence
    kernel_size
        Size of the convolutional kernel (default: 2)

    """

    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        kernel_size: int = 2,
    ):
        super().__init__()

        # Use 'same' padding to maintain sequence length through convolutions
        self.conv1 = nn.Conv1d(
            input_channels,
            2,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.activation1 = nn.Tanh()

        self.conv2 = nn.Conv1d(
            2,
            2,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.activation2 = nn.Tanh()

        self.pool = nn.MaxPool1d(kernel_size=10, stride=10)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process one-hot encoded amino acid features."""
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x


class BatchedHeads(nn.Module):
    """
    Parallel output heads sharing a hidden projection.

    Each head maps the shared trunk output to a scalar via a two-step
    computation: a batched linear projection followed by a per-head dot
    product with a learned weight vector.

    Parameters
    ----------
    input_size
        Size of the input feature vector (output of shared trunk).
    n_heads
        Number of parallel output heads.
    hidden
        Hidden dimension per head (default: 32).

    """

    def __init__(self, input_size: int, n_heads: int, hidden: int = 32):
        super().__init__()
        self.layer1 = nn.Linear(input_size, n_heads * hidden)
        self.w2 = nn.Parameter(torch.zeros(n_heads, hidden))
        self.b2 = nn.Parameter(torch.zeros(n_heads))
        nn.init.normal_(self.w2, std=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)
        n_heads = self.b2.shape[0]
        h = torch.relu(h.view(h.shape[0], n_heads, h.shape[1] // n_heads))
        return (h * self.w2.unsqueeze(0)).sum(dim=-1) + self.b2  # (batch, n_heads)


class DeepLCModel(nn.Module):
    """
    DeepLC model for peptide retention time prediction.

    Four parallel input branches — per-position atomic composition CNN, summed atomic
    composition CNN, global feature dense network, and one-hot amino acid CNN — are
    concatenated and passed through a shared dense trunk. Outputs are projected by
    :class:`BatchedHeads` to ``[batch, n_heads]``.

    When ``n_heads > 1`` the model is a multitask backbone trained across multiple LC
    setups. Call :meth:`add_adapter` to attach a fine-tuning MLP that maps the head
    vector to a single RT value ``[batch, 1]``.


    Parameters
    ----------
    n_heads
        Number of parallel output heads (default: 1).
    atom_sequence_length
        Length of the atomic composition sequence (default: 60)
    atom_channels
        Number of atomic feature channels (default: 6 for C,H,N,O,S,P)
    atom_sum_sequence_length
        Length of the summed atomic composition sequence (default: 30)
    global_feature_size
        Size of the global feature vector (default: 55)
    one_hot_sequence_length
        Length of the one-hot encoded sequence (default: 60)
    one_hot_channels
        Number of amino acid types for one-hot encoding (default: 20)
    one_hot_kernel_size
        Kernel size for one-hot branch convolutions (default: 2)
    atom_cnn_blocks
        Number of convolutional blocks in the atomic branch (default: 3)
    atom_cnn_kernel_size
        Kernel size for atomic branch convolutions (default: 5)
    atom_cnn_filters_start
        Starting number of filters in atomic branch (default: 256)
    atom_cnn_pool_size
        Max pooling size for atomic branch (default: 2)
    sum_cnn_blocks
        Number of convolutional blocks in the summed atomic branch (default: 3)
    sum_cnn_kernel_size
        Kernel size for summed atomic branch convolutions (default: 5)
    sum_cnn_filters_start
        Starting number of filters in summed atomic branch (default: 256)
    global_layer_size
        Layer size for global feature layers (default: 64)
    global_num_layers
        Number of dense layers in global branch (default: 4)
    final_layer_size
        Layer size for final dense layers (default: 128)
    final_num_layers
        Number of final dense layers (default: 5)
    regularizer_val
        L1 regularization coefficient (default: 0.000005)

    """

    def __init__(
        self,
        n_heads: int = 1,
        atom_sequence_length: int = 60,
        atom_channels: int = 6,
        atom_sum_sequence_length: int = 30,
        global_feature_size: int = 55,
        one_hot_sequence_length: int = 60,
        one_hot_channels: int = 20,
        one_hot_kernel_size: int = 2,
        atom_cnn_blocks: int = 3,
        atom_cnn_kernel_size: int = 5,
        atom_cnn_filters_start: int = 256,
        atom_cnn_pool_size: int = 2,
        sum_cnn_blocks: int = 3,
        sum_cnn_kernel_size: int = 5,
        sum_cnn_filters_start: int = 256,
        global_layer_size: int = 64,
        global_num_layers: int = 4,
        final_layer_size: int = 128,
        final_num_layers: int = 4,
        regularizer_val: float = 0.000005,
    ):
        super().__init__()

        # Branch A: Atomic composition CNN
        a_layers: list[nn.Module] = []
        in_channels = atom_channels
        for block_idx in range(atom_cnn_blocks):
            out_channels = int(atom_cnn_filters_start / (2**block_idx))
            use_pooling = block_idx < (atom_cnn_blocks - 1)
            a_layers.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    atom_cnn_kernel_size,
                    use_pooling=use_pooling,
                    pool_size=atom_cnn_pool_size,
                    regularizer_val=regularizer_val,
                )
            )
            in_channels = out_channels
        self.branch_a = nn.Sequential(*a_layers, nn.Flatten())

        # Branch B: Summed atomic composition CNN
        b_layers: list[nn.Module] = []
        in_channels = atom_channels
        for block_idx in range(sum_cnn_blocks):
            out_channels = int(sum_cnn_filters_start / (2**block_idx))
            use_pooling = block_idx < (sum_cnn_blocks - 1)
            b_layers.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    sum_cnn_kernel_size,
                    use_pooling=use_pooling,
                    pool_size=2,
                    regularizer_val=regularizer_val,
                )
            )
            in_channels = out_channels
        self.branch_b = nn.Sequential(*b_layers, nn.Flatten())

        # Branch C: Global features
        self.branch_c = GlobalFeatureBranch(
            global_feature_size,
            num_layers=global_num_layers,
            hidden_size=global_layer_size,
            regularizer_val=regularizer_val,
        )

        # Branch D: One-hot encoding
        self.branch_d = OneHotBranch(
            one_hot_channels, one_hot_sequence_length, kernel_size=one_hot_kernel_size
        )

        # Compute concatenated feature size via a dummy forward pass
        with torch.no_grad():
            concat_size = (
                self.branch_a(torch.zeros(1, atom_channels, atom_sequence_length)).shape[1]
                + self.branch_b(torch.zeros(1, atom_channels, atom_sum_sequence_length)).shape[1]
                + self.branch_c(torch.zeros(1, global_feature_size)).shape[1]
                + self.branch_d(torch.zeros(1, one_hot_channels, one_hot_sequence_length)).shape[1]
            )

        # Shared trunk: dense layers without output linear
        trunk_layers: list[nn.Module] = []
        for i in range(final_num_layers):
            in_features = concat_size if i == 0 else final_layer_size
            trunk_layers.extend([nn.Linear(in_features, final_layer_size), LeakyReLUSaturation()])
        self.shared_trunk = nn.Sequential(*trunk_layers)

        # Output heads
        self.heads = BatchedHeads(final_layer_size, n_heads)

        # Optional fine-tuning adapter (None until add_adapter() is called)
        self.adapter: nn.Module | None = None

        self._initialize_weights()

    def add_adapter(self, hidden_size: int = 256) -> None:
        """
        Attach a fine-tuning adapter mapping the head vector to one RT output.

        Adapter parameters are left at PyTorch default initialization and trained from scratch
        during fine-tuning.
        """
        n_heads = self.heads.b2.shape[0]
        self.adapter = nn.Sequential(
            nn.Linear(n_heads, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, max(1, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(1, hidden_size // 2), 1),
        )
        self.adapter.to(self.heads.b2.device)

    def freeze_backbone(self) -> None:
        """Freeze all parameters except the adapter."""
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith("adapter.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x_atom: torch.Tensor,
        x_atom_sum: torch.Tensor,
        x_global: torch.Tensor,
        x_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns
        -------
        torch.Tensor
            Shape ``[batch, n_heads]``, or ``[batch, 1]`` when an adapter is attached.

        """
        x_atom = x_atom.transpose(1, 2)
        x_atom_sum = x_atom_sum.transpose(1, 2)
        x_one_hot = x_one_hot.transpose(1, 2)
        concatenated = torch.cat(
            [
                self.branch_a(x_atom),
                self.branch_b(x_atom_sum),
                self.branch_c(x_global),
                self.branch_d(x_one_hot),
            ],
            dim=1,
        )
        out = self.heads(self.shared_trunk(concatenated))  # [batch, n_heads]
        adapter = getattr(self, "adapter", None)
        if adapter is not None:
            return adapter(out)  # [batch, 1]
        return out
