import logging
import GPUtil
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import subprocess
import hashlib
import os
import json
import datetime
from torchvision.datasets import ImageFolder
import argparse
from tqdm import tqdm
import copy

def get_max_batch_size(dataset_name, cuda_core=0):

    # Get GPU memory information
    gpus = GPUtil.getGPUs()
    if len(gpus) == 0:
        raise RuntimeError("No GPU found.")
    gpu = gpus[cuda_core]
    total_memory = gpu.memoryTotal * 1024 * 1024  # Convert to bytes
    reserved_memory = 0.1 * total_memory  # Reserve 10% of memory for other processes
    available_memory = total_memory - reserved_memory

    # Define common transformations including resizing
    resize_transform = transforms.Resize((32, 32))
    to_tensor_transform = transforms.ToTensor()

    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == "svhn":
        transform = transforms.Compose([
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset name")

    # Get a single sample to estimate memory usage
    sample = dataset[0][0].unsqueeze(0).to('cuda')
    sample_memory = sample.element_size() * sample.nelement()

    # Calculate the maximum batch size
    max_batch_size = available_memory // sample_memory
    max_batch_size = min(max_batch_size, len(dataset))  # Ensure batch size is not greater than dataset length
    return int(max_batch_size)


def clear_gpu_memory():
        logging.info("Clearing GPU memory.")
        try:
            # This command clears the GPU memory
            subprocess.run(['nvidia-smi', '--gpu-reset'], check=True)
            logging.info("Successfully cleared GPU memory.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to clear GPU memory: {e}")
        except Exception as e:
            logging.error(f"Unexpected error while clearing GPU memory: {e}")

class Reporter:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        hyperparameters_str = json.dumps(hyperparameters, sort_keys=True)
        hash_object = hashlib.md5(hyperparameters_str.encode())
        hash_hex = hash_object.hexdigest()[:8]
        self.log_filename = os.path.join("reports", f"R{hash_hex}.txt")
        self.ensure_log_file()

    def ensure_log_file(self):
        # Create the reports directory if it doesn't exist
        if not os.path.exists("reports"):
            os.makedirs("reports")
        
        # Remove the report file if it already exists
        if os.path.exists(self.log_filename):
            os.remove(self.log_filename)
        # Create or clean the log file
        with open(self.log_filename,'a') as f:
            # Write the initial process information and hyperparameters
            pid = os.getpid()
            f.write(f"Process ID: {pid}\n")
            f.write(f"Status: alive\n")
            f.write(f"Hyperparameters: {json.dumps(self.hyperparameters)}\n")
            f.write("\n")

    def log(self, message):
        with open(self.log_filename, 'a') as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{current_time}] {message}\n")

    def terminate(self):
        # Log process ended
        with open(self.log_filename, 'a') as f:
            self.log("Process ended\n")
        
        # Update the second line from "Status: alive" to "Status: dead"
        with open(self.log_filename, 'r') as f:
            lines = f.readlines()
        
        if len(lines) > 1 and "Status: alive" in lines[1]:
            lines[1] = "Status: dead\n"
        
        with open(self.log_filename, 'w') as f:
            f.writelines(lines)


def generate_data(dataset_name):
    # Define common transformations including resizing
    resize_transform = transforms.Resize((32, 32))
    to_tensor_transform = transforms.ToTensor()
    
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        return train_set  # Return the dataset directly
    
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        return train_set  # Return the dataset directly
    
    elif dataset_name == "cifar100":
        transform = transforms.Compose([
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        return train_set  # Return the dataset directly
    
    elif dataset_name == "svhn":
        transform = transforms.Compose([
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        return train_set  # Return the dataset directly
    
    elif dataset_name == "imagenet":
        transform = transforms.Compose([
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_set = ImageFolder(root='data/imagenet_data/data/imagenet', transform=transform)
        return train_set  # Return the dataset directly
    
    else:
        print("Unknown dataset name")
        raise ValueError("Unknown dataset name")


def copy_block(source_model, target_model, block_index):
    """
    Copies the parameters of a specific block from the source model to the target model.
    
    Args:
        source_model: The model from which to copy the parameters.
        target_model: The model to which the parameters will be copied.
        block_index: The index of the block whose parameters are to be copied.
    """
    # Ensure the block index is within the range of available blocks
    if block_index < 0 or block_index >= len(source_model.blocks):
        raise IndexError("Block index is out of range.")
    
    # Get the specific block from both models
    source_block = source_model.blocks[block_index]
    target_block = target_model.blocks[block_index]
    
    # Iterate over the parameters of the source block
    with torch.no_grad():  # Ensure no gradients are computed during parameter copying
        for source_param, target_param in zip(source_block.parameters(), target_block.parameters()):
            # Copy the parameter data to the target model's block
            # .data ensures we are modifying the tensor in-place and detach() ensures no gradient history is copied
            target_param.data = copy.deepcopy(source_param.data).detach()


def parse_arguments(H):
    """
    Dynamically parses command-line arguments based on the fields of an input dictionary H,
    and updates H with any new values provided via the command line.
    
    Args:
        H (dict): Initial dictionary containing default values for arguments.
    
    Returns:
        H (dict): Updated dictionary containing the parsed command-line arguments.
    """
    # Create the parser
    parser = argparse.ArgumentParser(description="Dynamically process input arguments for training configuration.")
    
    # Dynamically add arguments based on the keys and values of H
    for key, value in H.items():
        # Determine the type of the default value
        arg_type = type(value)
        if arg_type is bool:
            # Special handling for boolean flags
            parser.add_argument(f"--{key}", type=lambda x: (str(x).lower() == 'true'), default=value, help=f"{key} (default: {value})")
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=value, help=f"{key} (default: {value})")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Update H with arguments provided via command line
    H.update(vars(args))
    
    return H



def copy_model(source_model, target_model,device):
    """
    Copies the parameters from source_model to target_model.

    Parameters:
    - source_model: A PyTorch model from which to copy the parameters.
    - target_model: A PyTorch model to which the parameters will be copied.

    Both models should be of the same architecture.
    """
    # Ensure that both models are on the same device to avoid unnecessary data transfers
    target_model.to(device)
    # Get the state dictionary of the source model
    source_state_dict = source_model.state_dict()
    # Load the source state dictionary into the target model
    target_model.load_state_dict(source_state_dict)