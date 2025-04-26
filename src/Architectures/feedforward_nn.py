import torch
import torch.nn as nn
import torch.nn.functional as F

# Initial block for feedforward network: maps input vector to the first hidden layer
class InitialBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super(InitialBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

# Block for the hidden layers: each block is a single matrix multiplication with activation
class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

# Final block: maps the last hidden layer to the output logits, with an optional activation function
class FinalBlock(nn.Module):
    def __init__(self, in_features, output_dim, activation=None):
        super(FinalBlock, self).__init__()
        self.linear = nn.Linear(in_features, output_dim)
        self.activation = activation
    
    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

# FeedForwardNetwork model using a list of blocks.
class FeedForwardNetwork(nn.Module):
    def __init__(self, config, input_dim, output_dim, activation, final_activation=None):
        """
        Constructs a feedforward neural network with one block per matrix multiplication.

        Args:
            config (list of int): Each integer specifies the number of units in a hidden layer.
                                  Each element corresponds to one Linear block.
            in_features (int): The dimensionality of the input features.
            num_classes (int): The number of output classes.
            activation (callable): The activation function to use in each block.
            final_activation (callable, optional): The activation function to use in the final block.
        """
        super(FeedForwardNetwork, self).__init__()
        self.blocks = nn.ModuleList()

        # Add the initial block that maps input features to the first hidden layer
        self.blocks.append(InitialBlock(input_dim, config[0], activation))

        # Add hidden blocks: each block performs a linear transformation with an activation
        for i in range(1, len(config)):
            self.blocks.append(LinearBlock(config[i - 1], config[i], activation))
        
        # Add the final block mapping the last hidden layer to the output classes
        self.blocks.append(FinalBlock(config[-1], output_dim, final_activation))
    
    def forward(self, x):
        """
        Forward pass: sequentially apply each block in the network.
        """
        for block in self.blocks:
            x = block(x)
        return x
    


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = activation
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x

# -----------------------------------------------------------------------------
# Residual Convolutional Block
# -----------------------------------------------------------------------------


class ResidualConvBlock(nn.Module):
    """A convolutional block with a weighted residual (skip) connection.

    The output is computed as

        y = f(x) + g(x)

    where *f* is a 3×3 convolution followed by *activation* and *g* is either
    the identity (if ``in_channels == out_channels``) or a 1×1 convolution used
    to match the channel dimension. 
    """

    def __init__(self, in_channels, out_channels, activation):
        super().__init__()


        self.activation = activation

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(in_channels)


        # Fix: Add skip connection handling
        self.skip_connection = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        out = self.bn(x)
        out = self.activation(self.conv1(out))
        out = self.activation(self.conv2(out))
        skip = self.skip_connection(x)
        return out + skip
    
# Final block: maps the last hidden layer to the output logits, with an optional activation function
class FinalBlockCnn(nn.Module):
    def __init__(self, in_channels, output_dim, final_activation):
        super(FinalBlockCnn, self).__init__()
        self.pool= nn.AdaptiveAvgPool2d((1, 1))
        self.nn = nn.Linear(in_channels, output_dim)
        self.final_activation = final_activation
    
    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.nn(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x

class FeedForwardCNN(nn.Module):
    def __init__(self, config, in_channels, output_dim, activation, final_activation=None):
        """
        Constructs a feedforward CNN with convolutional blocks similar to a standard NN.

        Args:
            config (list of int): Number of output channels for each convolutional block.
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
            output_dim (int): Number of output units (e.g., number of classes).
            activation (callable): Activation function for convolutional blocks.
            final_activation (callable, optional): Activation for the final output.
        """
        super(FeedForwardCNN, self).__init__()
        self.blocks = nn.ModuleList()
        
        # Initial convolutional block
        self.blocks.append(ConvBlock(in_channels, config[0], activation))
        
        # Hidden convolutional blocks
        for i in range(1, len(config)):
            self.blocks.append(ConvBlock(config[i - 1], config[i], activation))
        
        self.blocks.append(FinalBlockCnn(config[-1],output_dim,final_activation))

    def forward(self, x):
        """
        Forward pass: applies convolutional blocks, pooling, and final linear transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        for block in self.blocks:
            x = block(x)
        return x


# -----------------------------------------------------------------------------
# Residual Feed‑Forward CNN
# -----------------------------------------------------------------------------


class ResidualFeedForwardCNN(nn.Module):
    """Feed‑forward CNN where each block includes a weighted residual skip.

    Parameters
    ----------
    config : list[int]
        Output channels of each residual convolution block.
    in_channels : int
        Input image channels.
    output_dim : int
        Dimensionality of the output (e.g. number of classes).
    activation : callable
        Activation applied after each convolution.
        Weight of the main path output vs. the skip connection. Must be in
        [0, 1].  ``0`` reduces to pure skip (identity) while ``1`` recovers the
        vanilla feed‑forward CNN.
    final_activation : callable | None, default = None
        Activation for the final linear layer.
    """

    def __init__(
        self,
        config,
        in_channels,
        output_dim,
        activation,
        final_activation=None,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        # Initial residual block

        self.blocks.append(nn.Sequential(
            nn.Conv2d(
                in_channels,
                config[0],
                kernel_size=1,
                padding=1,
                bias=True
            ),
            nn.ReLU()
        ))

        # Subsequent residual blocks
        for i in range(1, len(config)):
            self.blocks.append(
                ResidualConvBlock(config[i - 1], config[i], activation)
            )

        # Final classification block
        self.blocks.append(FinalBlockCnn(config[-1], output_dim, final_activation))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ======================================================================================
# Ensemble of Feed‑Forward Networks
# ======================================================================================


class EnsembleFeedForwardNetwork(nn.Module):
    """Ensemble that averages the outputs of several ``FeedForwardNetwork`` models.

    Parameters
    ----------
    configs : list[list[int]]
        A *2‑D* list where each inner list contains the hidden sizes for one
        sub‑model. Example: ``[[128, 64], [256, 128, 64]]`` creates two
        feed‑forward networks.
    input_dim : int
        Dimensionality of the input features.
    output_dim : int
        Dimensionality of the output (number of classes or regression targets).
    activation : callable
        Activation function used in hidden layers.
    final_activation : callable | None, default = None
        Activation function for the final layer.
    """

    def __init__(self, configs, input_dim, output_dim, activation, final_activation=None,
                 voting_method: str = "random"):
        super().__init__()

        # Validate *configs*
        if not isinstance(configs, (list, tuple)) or len(configs) == 0:
            raise ValueError("`configs` must be a non‑empty 2‑D list of configurations.")

        for cfg in configs:
            if not isinstance(cfg, (list, tuple)) or len(cfg) == 0:
                raise ValueError(
                    "Each element of `configs` must be a non‑empty list/tuple of integers."
                )

        # Build sub‑models
        self.voting_method = voting_method.lower()

        if self.voting_method not in {"average", "random"}:
            raise ValueError("voting_method must be 'average' or 'random'")

        self.models = nn.ModuleList(
            [
                FeedForwardNetwork(
                    list(cfg),  # ensure each cfg is a list, copy to avoid mutations
                    input_dim,
                    output_dim,
                    activation,
                    final_activation,
                )
                for cfg in configs
            ]
        )

        # Concatenate blocks from all sub‑models to keep compatibility with utilities
        # that expect a single `blocks` attribute.
        self.blocks = nn.ModuleList()
        for model in self.models:
            self.blocks.extend(model.blocks)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        """Return ensemble prediction according to *voting_method*."""

        outputs = [m(x) for m in self.models]
        out_stack = torch.stack(outputs, dim=0)  # (E, B, D)

        if self.voting_method == "average":
            return out_stack.mean(dim=0)

        # Random voting: choose one sub‑model per sample uniformly at random.
        E, B, _ = out_stack.shape
        rand_idx = torch.randint(0, E, (B,), device=out_stack.device)
        mask = torch.nn.functional.one_hot(rand_idx, num_classes=E).float()  # (B, E)
        mask = mask.permute(1, 0).unsqueeze(-1)  # (E, B, 1)
        selected = (out_stack * mask).sum(dim=0)  # (B, D)
        return selected


# ======================================================================================
# Ensemble of Feed‑Forward CNNs
# ======================================================================================


class EnsembleFeedForwardCNN(nn.Module):
    """Ensemble that averages outputs from several ``FeedForwardCNN`` models.

    Parameters
    ----------
    configs : list[list[int]]
        A 2‑D list where each inner list contains the out‑channels for the
        convolutional blocks of one sub‑model.
    in_channels : int
        Number of channels of the input images.
    output_dim : int
        Dimensionality of the output (e.g. number of classes).
    activation : callable
        Activation function for hidden conv blocks.
    final_activation : callable | None
        Activation for the final fully‑connected layer.
    """

    def __init__(self, configs, in_channels, output_dim, activation, final_activation=None,
                 voting_method: str = "random"):
        super().__init__()

        # Validate configs
        if not isinstance(configs, (list, tuple)) or len(configs) == 0:
            raise ValueError("`configs` must be a non‑empty 2‑D list of configurations.")
        for cfg in configs:
            if not isinstance(cfg, (list, tuple)) or len(cfg) == 0:
                raise ValueError("Each configuration in `configs` must be a non‑empty list/tuple.")

        # Build sub‑models
        self.voting_method = voting_method.lower()

        if self.voting_method not in {"average", "random"}:
            raise ValueError("voting_method must be 'average' or 'random'")

        self.models = nn.ModuleList(
            [
                FeedForwardCNN(
                    list(cfg),  # ensure copy
                    in_channels,
                    output_dim,
                    activation,
                    final_activation,
                )
                for cfg in configs
            ]
        )

        # Flatten blocks of all sub‑models
        self.blocks = nn.ModuleList()
        for m in self.models:
            self.blocks.extend(m.blocks)

    def forward(self, x):
        outputs = [m(x) for m in self.models]
        out_stack = torch.stack(outputs, dim=0)  # (E, B, D)

        if self.voting_method == "average":
            return out_stack.mean(dim=0)

        # Random voting (uniform) per sample
        E, B, _ = out_stack.shape
        rand_idx = torch.randint(0, E, (B,), device=out_stack.device)
        mask = torch.nn.functional.one_hot(rand_idx, num_classes=E).float().permute(1, 0).unsqueeze(-1)
        return (out_stack * mask).sum(dim=0)