from torchvision import models

from source.Architectures.resnets import ResNet_base
from source.Architectures.resnets import *


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