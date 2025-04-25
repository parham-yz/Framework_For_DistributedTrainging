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
        else:
            # raise Exception("ResNet Arc, error!")
            identity = x
        
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


# -------------------------------------------------------------------------
# NEW CLASS: ResNet_base_bi_partitioned
# -------------------------------------------------------------------------
class ResNet_base_bi_partitioned(nn.Module):
    """
    Modified ResNet base model that partitions all its blocks into exactly two sequential ModuleLists.
    The partitioning aims for roughly half the blocks in each list.
    """
    def __init__(self, config, num_classes=1000):
        super(ResNet_base_bi_partitioned, self).__init__()

        # Temporary list to hold all blocks before partitioning
        all_blocks = []

        # Add the initial block (conv1, bn1, maxpool)
        initial_block = InitialBlock(in_channels=3, out_channels=64)
        all_blocks.append(initial_block)
        current_in_channels = 64

        # Add BasicBlocks based on the config
        for out_channels, num_blocks in config:
            # Determine stride and downsampling for the first block of this layer/stage
            stride = 1
            downsample = None
            # Check if stride needs to be 2 (feature map size reduction) or if channels change
            if current_in_channels != out_channels * BasicBlock.expansion:
                 stride = 2
                 downsample = nn.Sequential(
                     nn.Conv2d(current_in_channels, out_channels * BasicBlock.expansion,
                               kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(out_channels * BasicBlock.expansion),
                 )
            elif stride != 1: # This condition might be redundant if the first check covers channel changes, but keep for safety
                 downsample = nn.Sequential(
                     nn.Conv2d(current_in_channels, out_channels * BasicBlock.expansion,
                               kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(out_channels * BasicBlock.expansion),
                 )


            # Append the first block of the current layer/stage
            all_blocks.append(BasicBlock(current_in_channels, out_channels, stride, downsample))
            current_in_channels = out_channels * BasicBlock.expansion # Update input channels for the next block

            # Append the remaining blocks in this layer/stage (stride=1, no downsample needed within the layer)
            for _ in range(1, num_blocks):
                all_blocks.append(BasicBlock(current_in_channels, out_channels, stride=1, downsample=None))

        # Add the final block (avgpool, fc)
        # Note: For ResNet18, the expected in_features before avgpool is 512
        final_block = FinalBlock(in_features=current_in_channels, num_classes=num_classes)
        all_blocks.append(final_block)

        # Partition the collected blocks into two roughly equal halves
        num_total_blocks = len(all_blocks)
        split_point = num_total_blocks // 2  # Integer division for the split index

        self.blocks_part1 = nn.ModuleList(all_blocks[:split_point])
        self.blocks_part2 = nn.ModuleList(all_blocks[split_point:])
        self.blocks = [self.blocks_part1, self.blocks_part2]
        # Sanity check (optional)
        # print(f"Total blocks: {num_total_blocks}")
        # print(f"Partition 1 size: {len(self.blocks_part1)}")
        # print(f"Partition 2 size: {len(self.blocks_part2)}")


    def forward(self, x):
        # Apply blocks in the first partition
        for block in self.blocks_part1:
            x = block(x)

        # Apply blocks in the second partition
        for block in self.blocks_part2:
            x = block(x)

        return x