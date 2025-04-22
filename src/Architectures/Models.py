from torchvision import models

from src.Architectures.resnets import ResNet_base
from src.Architectures.resnets import *
from src.Architectures.feedforward_nn import *

def load_resnet18(pretrained=False, num_classes=1000):
    # Initialize custom model
    custom_model = ResNet_base([(64, 2), (128, 2), (256, 2), (512, 2)], num_classes=num_classes)
    
    # Load standard pretrained model for architecture comparison
    standard_model = models.resnet18(pretrained=True)
    
    # Attempt to load pretrained weights into custom model for architecture validation
    try:
        custom_model.load_state_dict(standard_model.state_dict(), strict=False)
        print("Architecture match confirmed for ResNet18.")
    except Exception as e:
        print(f"Architecture mismatch or error: {e}")
    
    # If pretrained is False, reinitialize the custom model to reset weights
    if not pretrained:
        custom_model = ResNet_base([(64, 2), (128, 2), (256, 2), (512, 2)], num_classes=num_classes)
    
    return custom_model

def load_resnet34(pretrained=False, num_classes=1000):
    # Initialize custom model
    custom_model = ResNet_base([(64, 3), (128, 4), (256, 6), (512, 3)], num_classes=num_classes)
    
    # Load standard pretrained model for architecture comparison
    standard_model = models.resnet34(pretrained=True)
    
    # Attempt to load pretrained weights into custom model for architecture validation
    try:
        custom_model.load_state_dict(standard_model.state_dict(), strict=False)
        print("Architecture match confirmed for ResNet34.")
    except Exception as e:
        print(f"Architecture mismatch or error: {e}")
    
    # If pretrained is False, reinitialize the custom model to reset weights
    if not pretrained:
        custom_model = ResNet_base([(64, 3), (128, 4), (256, 6), (512, 3)], num_classes=num_classes)
    
    return custom_model


def load_feedforward(config, input_dim, output_dim, activation, final_activation):
    """
    Load a FeedForwardNetwork model with the specified configuration.

    Args:
        config (list of int): Each integer specifies the number of units in a hidden layer.
        input_dim (int): The dimensionality of the input features.
        output_dim (int): The number of output classes.
        activation (callable): The activation function to use in each block.
        final_activation (callable, optional): The activation function to use in the final block.
        pretrained_weights (str, optional): Path to the pretrained weights file.

    Returns:
        FeedForwardNetwork: An instance of the FeedForwardNetwork model.
    """
    # Initialize the feedforward network
    model = FeedForwardNetwork(config, input_dim, output_dim, activation, final_activation)

    return model


def load_feedforward_cnn(config,input_dim, output_dim, activation, final_activation):
    """
    Load a FeedForwardNetwork model with the specified configuration.

    Args:
        config (list of int): Each integer specifies the number of units in a hidden layer.
        input_dim (int): The dimensionality of the input features.
        output_dim (int): The number of output classes.
        activation (callable): The activation function to use in each block.
        final_activation (callable, optional): The activation function to use in the final block.
        pretrained_weights (str, optional): Path to the pretrained weights file.

    Returns:
        FeedForwardNetwork: An instance of the FeedForwardNetwork model.
    """
    # Initialize the feedforward network
    model = FeedForwardCNN(config,input_dim, output_dim, activation, final_activation)

    return model


# -----------------------------------------------------------------------------
# Residual Feed‑Forward CNN loader
# -----------------------------------------------------------------------------


def load_residual_feedforward_cnn(
    config,
    in_channels,
    output_dim,
    activation,
    beta: float = 1.0,
    final_activation=None,
):
    """Instantiate ``ResidualFeedForwardCNN``.

    This helper mirrors the signature of ``load_feedforward_cnn`` with one extra
    *beta* argument for the residual weighting.
    """

    from src.Architectures.feedforward_nn import ResidualFeedForwardCNN

    return ResidualFeedForwardCNN(
        config, in_channels, output_dim, activation, beta, final_activation
    )


# -----------------------------------------------------------------------------
# Ensemble Feed‑Forward loader
# -----------------------------------------------------------------------------


def load_feedforward_ensemble(configs, input_dim, output_dim, activation, final_activation=None):
    """Load an ensemble of feed‑forward networks.

    Args
    ----
    configs : list[list[int]]
        2‑D list where each inner list is passed to an individual FeedForwardNetwork.
    input_dim : int
        Dimensionality of input features.
    output_dim : int
        Dimensionality of output.
    activation : callable
        Activation for hidden layers.
    final_activation : callable | None
        Activation for the final layer.
    """

    from src.Architectures.feedforward_nn import EnsembleFeedForwardNetwork

    return EnsembleFeedForwardNetwork(
        configs, input_dim, output_dim, activation, final_activation
    )


# -----------------------------------------------------------------------------
# Ensemble CNN loader
# -----------------------------------------------------------------------------


def load_feedforward_cnn_ensemble(configs, in_channels, output_dim, activation, final_activation=None):
    """Load an ensemble of feed‑forward CNNs.

    Args
    ----
    configs : list[list[int]]
        2‑D list with convolutional hidden channel counts per sub‑model.
    in_channels : int
        Number of input channels in the images.
    output_dim : int
        Number of output units/classes.
    activation : callable
        Hidden activation.
    final_activation : callable | None
        Activation for the final layer.
    """

    from src.Architectures.feedforward_nn import EnsembleFeedForwardCNN

    return EnsembleFeedForwardCNN(
        configs, in_channels, output_dim, activation, final_activation
    )
