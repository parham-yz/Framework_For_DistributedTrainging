import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import os
import copy
import time
import json
import sys
import torch.utils
import signal
import sys
import fcntl
from Gans import *
from Inception import*
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils
import itertools
from data_imagenet import ImageFolder

# Hyperparameters
H = {
    "batch_size": 256,
    "hidden_dim": 128,
    "step_size": 0.00001,
    "evaluation_samples": 256,
    "rounds": 20000,  # Changed from epochs to rounds
    "K": 1,
    "dataset_name": "mnist",  # Default dataset
    "cuda_core": 0,  # Default CUDA core
}

Reporter = None

def generate_data(dataset_name="mnist"):
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
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        # Extract inputs only
        inputs = [data[0] for data in train_set]
        return torch.stack(inputs)
    
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        # Extract inputs only
        inputs = [data[0] for data in train_set]
        return torch.stack(inputs)
    
    elif dataset_name == "svhn":
        transform = transforms.Compose([
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        # Extract inputs only
        inputs = [data[0] for data in train_set]
        return torch.stack(inputs)
    
    elif dataset_name == "imagenet":
        transform = transforms.Compose([
            resize_transform,
            to_tensor_transform,
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_set = ImageFolder(root='data/imagenet_data/data/imagenet', transform=transform)
        # Extract inputs only
        inputs = [data[0] for data in train_set]
        return torch.stack(inputs)
    
    else:
        Reporter.log("Unknown dataset name")
        raise ValueError("Unknown dataset name")        

def compute_mmd(x, y, kernel='rbf', sigma=1.0):
    x, y = x.view(x.shape[0], -1), y.view(y.shape[0], -1)
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    if kernel == 'rbf':
        K = torch.exp(-sigma * (rx.t() + rx - 2 * xx))
        L = torch.exp(-sigma * (ry.t() + ry - 2 * yy))
        P = torch.exp(-sigma * (rx.t() + ry - 2 * zz))
    else:
        Reporter.log("Unknown kernel type.")
        raise ValueError('Unknown kernel type.')

    beta = (1. / (x.size(0) * x.size(0)))
    gamma = (1. / (y.size(0) * y.size(0)))
    delta = (2. / (x.size(0) * y.size(0)))

    return beta * torch.sum(K) + gamma * torch.sum(L) - delta * torch.sum(P)


class GAN:
    def __init__(self, H):
        self.hidden_dim = H["hidden_dim"]
        self.lr = H["step_size"]
        self.batch_size = H["batch_size"]
        self.rounds = H["rounds"]  # Changed from epochs to rounds
        self.K = H["K"]
        self.device = torch.device(f"cuda:{H['cuda_core']}")
        self.evaluation_samples = H["evaluation_samples"]
        self.loss_history = []
        
        # Print hyperparameters at the beginning if printing is enabled
        Reporter.log(f"K: {self.K}, Step Size: {self.lr}, CUDA Core: {H['cuda_core']}, Dataset: {H['dataset_name']}, Rounds: {self.rounds}")
        # Determine model type based on dataset
        if H["dataset_name"] == "mnist":
            self.generator = Generator_Mnist().to(self.device)
            self.discriminator = Discriminator_Mnist().to(self.device)
            self.z_dim = [H["batch_size"], 5, 1, 1]  # Set z_dim for MNIST
            self.label_size = [H["batch_size"], 1, 1, 1]  # Set label_size for MNIST
        elif H["dataset_name"] == "cifar10":
            self.generator = Generator_ImageNet(z_dim=256).to(self.device)  # Use ImageNet generator for CIFAR-10
            self.discriminator = Discriminator_ImageNet().to(self.device)  # Use ImageNet discriminator for CIFAR-10
            self.z_dim = [H["batch_size"], 256]  # Set z_dim for CIFAR-10
            self.label_size = [H["batch_size"], 1]  # Set label_size for CIFAR-10
        elif H["dataset_name"] == "svhn":
            self.generator = Generator_CIFAR().to(self.device)  # Assuming CIFAR generator is used for SVHN
            self.discriminator = Discriminator_CIFAR().to(self.device)  # Assuming CIFAR discriminator is used for SVHN
            self.z_dim = [H["batch_size"], 100, 1, 1]  # Set z_dim for SVHN
            self.label_size = [H["batch_size"], 1, 1, 1]  # Set label_size for SVHN
        elif H["dataset_name"] == "imagenet":
            self.generator = Generator_ImageNet(z_dim=256).to(self.device)
            self.discriminator = Discriminator_ImageNet().to(self.device)
            self.z_dim = [H["batch_size"], 256]  # Corrected z_dim for ImageNet
            self.label_size = [H["batch_size"], 1]  # Set label_size for ImageNet
        else:
            Reporter.log("Unknown dataset name")
            raise ValueError("Unknown dataset name")
        
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self.criterion = nn.BCELoss()
        
        self.train_loader = torch.utils.data.DataLoader(
            generate_data(H["dataset_name"]),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

    def backprob_discriminator(self, real_samples, generator):
        batch_size = real_samples.size(0)
        real_labels = torch.ones(self.label_size).to(self.device)
        fake_labels = torch.zeros(self.label_size).to(self.device)
        self.optimizer_D.zero_grad()
        
        outputs = self.discriminator(real_samples)
        d_loss_real = self.criterion(outputs.view(-1), real_labels.view(-1))
        d_loss_real.backward()
        
        z = torch.randn(batch_size, *self.z_dim[1:]).to(self.device)
        fake_samples = generator(z)
        outputs = self.discriminator(fake_samples.detach())
        d_loss_fake = self.criterion(outputs, fake_labels)
        d_loss_fake.backward()
        
        return d_loss_real + d_loss_fake

    def backprob_generator(self, discriminator):
        self.optimizer_G.zero_grad()
        
        z = torch.randn(H["batch_size"], *self.z_dim[1:]).to(self.device)
        fake_samples = self.generator(z)
        outputs = discriminator(fake_samples)
        g_loss = self.criterion(outputs, torch.ones(outputs.shape).to(self.device))
        g_loss.backward()
        
        return g_loss

def train(gan):
    device = gan.device

    fid_calculator = FIDCalculator(device)
    start_time = time.time()  # Record the start time
    iteration = 0
    pupy_discriminator = type(gan.discriminator)().to(device)
    pupy_generator = type(gan.generator)(gan.generator.z_dim).to(device)
    
    try:
        Reporter.log("Started the training loop")
        while iteration < gan.rounds * gan.K:  # Changed from gan.epochs to gan.rounds
            for batch in gan.train_loader:
                
                if batch.size(0) != gan.train_loader.batch_size:
                    continue
                if batch.dim() == 3:
                    batch = batch.unsqueeze(1)
                batch = batch.to(device)
                

                if iteration % (gan.K * 200) == 0:
                    # Generate fake samples
                    z = torch.randn(gan.evaluation_samples, *gan.z_dim[1:]).to(device)
                    fake_samples = gan.generator(z)
                    
                    # Compute FID
                    with torch.no_grad():
                        fid = fid_calculator.calculate_fid(batch, fake_samples.detach())
                    loss = fid
                    
                    # Print progress
                    Reporter.log(f"Iteration {iteration}: FID Loss = {loss}")

                    # Update loss history
                    
                    gan.loss_history.append(loss)
                    
   
                
                if iteration % gan.K == 0:
                    # Create deep copies of the generator and discriminator
                    state_generator = copy.deepcopy(gan.generator.state_dict())
                    pupy_generator.load_state_dict(state_generator)

                    state_discriminator = copy.deepcopy(gan.discriminator.state_dict())
                    pupy_discriminator.load_state_dict(state_discriminator)
                    
                    # Train discriminator
                    gan.backprob_discriminator(batch, pupy_generator)
                    
                    # Train generator
                    gan.backprob_generator(pupy_discriminator)
                
                             # Step optimizers
                gan.optimizer_G.step()
                gan.optimizer_D.step()
                torch.cuda.empty_cache()  # Clear unused variables from GPU memory
                iteration += 1
                
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the execution time
        Reporter.log(f"Training completed in {execution_time:.2f} seconds")
    
    except Exception as e:
        Reporter.log(f"Training terminated due to an error: {e}")
        Reporter.terminate()
        raise

def update_progress_tracker(rounds, file):  # Changed from epochs to rounds
    pid = os.getpid()
    with open(file, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file for exclusive access
        try:
            data = json.load(f)
            data[str(pid)] = rounds  # Update the value by adding rounds
            f.seek(0)  # Move the cursor to the beginning of the file
            json.dump(data, f)  # Write the updated data back to the file
            f.truncate()  # Remove any leftover data from the previous write
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file after operation
            
def handle_interrupt(signal, frame):
    Reporter.log("Training interrupted by user")
    Reporter.terminate()
    sys.exit(0)

if __name__ == '__main__':
    
    
    if len(sys.argv) == 6:
        H["step_size"] = float(sys.argv[1])
        H["K"] = int(sys.argv[2])
        H["rounds"] = int(sys.argv[3])  # Changed from epochs to rounds
        H["dataset_name"] = sys.argv[4]
        H["cuda_core"] = int(sys.argv[5])
    elif len(sys.argv) != 1:
        raise ValueError("Invalid number of arguments provided")

    signal.signal(signal.SIGINT, handle_interrupt)

    try:
        Reporter= utils.Reporter(hyperparameters=H)
        gan = GAN(H)

        total_params = sum(p.numel() for p in itertools.chain(gan.generator.parameters(), gan.discriminator.parameters()))
        Reporter.log(f"Total number of parameters in the GAN model: {total_params}")
        # Train the GAN with the specified dataset and hyperparameters
        train(gan)

        # After training, you can log the data and plot MMD or other metrics if needed
        log_data = {
            "hyper_parameters": H,
            "loss_log": gan.loss_history
        }

        log_filename = f"lr={gan.lr}_K={gan.K}_dataset={H['dataset_name']}.json"
        log_path = "saved_logs/" + log_filename

        if not os.path.exists("saved_logs"):
            os.makedirs("saved_logs")

        with open(log_path, 'w') as f:
            json.dump(log_data, f)

        plt.plot(gan.loss_history)
        plt.title(f"Running Average FID Loss - {H['dataset_name']}")
        plt.xlabel("Iterations")
        plt.ylabel("FID Loss")
        plt.savefig(f"res_{H['dataset_name']}.pdf")
        Reporter.log(f"Training completed on dataset: {H['dataset_name']}")
        Reporter.terminate()
    
    except Exception as e:
        Reporter.log(f"Process terminated due to an error: {e}")
        Reporter.terminate()
        raise