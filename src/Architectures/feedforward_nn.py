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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = activation
    
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
    
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