"""
PyTorch architecture definitions for DeepLC.

This module contains the neural network architectures used by DeepLC for
predicting peptide retention times based on atomic composition and other features.
"""

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

        if use_pooling:
            self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

        # Store regularizer value for potential use in training
        self.regularizer_val = regularizer_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        if self.use_pooling:
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


class DeepLCModel(nn.Module):
    """
    Complete DeepLC model for peptide retention time prediction.

    This model consists of multiple branches processing different feature types:
    - Atomic composition CNN (per-position atomic features)
    - Summed atomic composition CNN (aggregated atomic features)
    - Global feature branch (peptide-level features)
    - One-hot encoding branch (amino acid sequence)

    The outputs are concatenated and passed through a deep fully connected network
    for final prediction.

    Parameters
    ----------
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
        final_num_layers: int = 5,
        regularizer_val: float = 0.000005,
    ):
        super().__init__()

        # Branch A: Atomic composition CNN
        a_layers = []
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
        b_layers = []
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
            one_hot_channels,
            one_hot_sequence_length,
            kernel_size=one_hot_kernel_size,
        )

        # Calculate concatenated feature size
        # Need to compute output sizes after convolutions and pooling
        with torch.no_grad():
            dummy_a = torch.zeros(1, atom_channels, atom_sequence_length)
            dummy_b = torch.zeros(1, atom_channels, atom_sum_sequence_length)
            dummy_c = torch.zeros(1, global_feature_size)

            out_a = self.branch_a(dummy_a)
            out_b = self.branch_b(dummy_b)
            out_c = self.branch_c(dummy_c)

            dummy_d = torch.zeros(1, one_hot_channels, one_hot_sequence_length)
            out_d = self.branch_d(dummy_d)

            concat_size = out_a.shape[1] + out_b.shape[1] + out_c.shape[1] + out_d.shape[1]

        # Final dense layers
        final_layers = []
        for i in range(final_num_layers):
            in_features = concat_size if i == 0 else final_layer_size
            final_layers.extend(
                [
                    nn.Linear(in_features, final_layer_size),
                    LeakyReLUSaturation(),
                ]
            )

        # Output layer
        final_layers.append(nn.Linear(final_layer_size, 1))

        self.final_network = nn.Sequential(*final_layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using normal distribution (matching TensorFlow's RandomNormal)."""
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
        Forward pass through the DeepLC model.

        Parameters
        ----------
        x_atom
            Atomic composition features [batch, sequence_length, channels]
        x_atom_sum
            Summed atomic composition features [batch, sequence_length, channels]
        x_global
            Global peptide features [batch, feature_size]
        x_one_hot
            One-hot encoded amino acid features [batch, sequence_length, channels]

        Returns
        -------
        torch.Tensor
            Predicted retention times [batch, 1]

        """
        # Transpose to Conv1D format: (batch, channels, length)
        x_atom = x_atom.transpose(1, 2)
        x_atom_sum = x_atom_sum.transpose(1, 2)
        x_one_hot = x_one_hot.transpose(1, 2)

        # Process each branch
        out_a = self.branch_a(x_atom)
        out_b = self.branch_b(x_atom_sum)
        out_c = self.branch_c(x_global)
        out_d = self.branch_d(x_one_hot)

        # Concatenate features
        concatenated = torch.cat([out_a, out_b, out_c, out_d], dim=1)

        # Final prediction
        output = self.final_network(concatenated)

        return output
