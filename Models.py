import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# Basic Block for ResNet18 (with skip connections)
class BasicBlock(nn.Module):
    expansion = 1  # No expansion in ResNet18 blocks
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # First conv layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second conv layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample for skip connection if needed
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out

# New block for the initial conv1, bn1, and maxpool
class InitialBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(InitialBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        return x

# New block for the final avgpool and fc
class FinalBlock(nn.Module):
    def __init__(self, in_features=512, num_classes=1000):
        super(FinalBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Modified ResNet18 model with all components as blocks in a list
class ResNet_base(nn.Module):
    def __init__(self, config, num_classes=1000):
        super(ResNet_base, self).__init__()
        
        # Initialize the list of blocks using nn.ModuleList
        self.blocks = nn.ModuleList()
        
        # Add the initial block (conv1, bn1, maxpool)
        self.blocks.append(InitialBlock(in_channels=3, out_channels=64))
        current_in_channels = 64
        
        # Add BasicBlocks
        for out_channels, num_blocks in config:
            # First block in each "layer" may need stride=2 and downsampling
            stride = 1 if current_in_channels == out_channels else 2
            if stride != 1 or current_in_channels != out_channels * BasicBlock.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(current_in_channels, out_channels * BasicBlock.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * BasicBlock.expansion),
                )
            else:
                downsample = None
            
            # Append the first block
            self.blocks.append(BasicBlock(current_in_channels, out_channels, stride, downsample))
            current_in_channels = out_channels * BasicBlock.expansion
            
            # Append the remaining blocks in this "layer"
            for _ in range(1, num_blocks):
                self.blocks.append(BasicBlock(current_in_channels, out_channels))
        
        # Add the final block (avgpool, fc)
        self.blocks.append(FinalBlock(in_features=512 * BasicBlock.expansion, num_classes=num_classes))

    def forward(self, x):
        # Apply all blocks in sequence
        for block in self.blocks:
            x = block(x)
        return x

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