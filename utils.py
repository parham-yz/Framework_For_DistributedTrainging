import logging
import GPUtil
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import subprocess
import hashlib


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
            transforms.Grayscale(num_output_channels=1),
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

import os
import json

import datetime

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
