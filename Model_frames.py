import torch
import torch.nn as nn
import torch.optim as optim
import copy
import Models 
import utils
import signal
import sys
import data
import gc

class ImageClassifier_frame_blockwise:
    def __init__(self, model, H):
        self.H = H
        self.lr = H["step_size"]
        self.batch_size = H["batch_size"]
        self.rounds = H["rounds"]
        self.K = H["K"]
        self.device = torch.device(f"cuda:{H['cuda_core']}")
        self.dataset_name = H["dataset_name"]
        self.loss_history = []
        self.param_deviation = []

        # Move the model to the specified device (e.g., GPU)
        self.center_model = model.to(self.device)

        self.distributed_models = {}
        self.optimizers = []

        # Iterate over the blocks of the model
        for i, block in enumerate(self.center_model.blocks):
            # Create a deep copy of the model for each block
            copy_model = copy.deepcopy(self.center_model)
            # Initialize an optimizer for the parameters of the current block
            optimizer = optim.Adam(copy_model.blocks[i].parameters(), lr=self.lr)
            self.optimizers.append(optimizer)
            # Map each block to its corresponding model and optimizer
            self.distributed_models[f"block_{i}"] = (copy_model, optimizer)

        # Define the loss function
        self.criterion = nn.CrossEntropyLoss()

        # Load the dataset into memory and transfer it to the device
        self.dataset = [(data[0].to(self.device), data[1]) for data in data.generate_imagedata(self.dataset_name)]
        # Determine the sizes of the training and test sets
        train_size = int(0.9 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        # Split the dataset into training and test sets
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])
        # Create data loaders for the training set
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        # Create a larger data loader for the training set (for different batch processing)
        self.big_train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size * 8,
            shuffle=True,
            drop_last=True
        )

        # Initialize the reporting mechanism and log additional information about frame type, number of blocks, and number of model parameters
        self.reporter = utils.Reporter(H)
        frame_type = type(self).__name__
        num_blocks = len(self.center_model.blocks)
        num_parameters = sum(p.numel() for p in self.center_model.parameters())
        self.reporter.log(f"Frame Type: {frame_type}, Number of Blocks: {num_blocks}, Number of Model Parameters: {num_parameters}")

    def communicate(self):
        # Synchronize the blocks with the central model
        for block_idx, block in enumerate(self.distributed_models.keys()):
            model, _ = self.distributed_models[block]
            utils.copy_block(model, self.center_model, block_idx)

        # Synchronize the central model with the blocks
        for block in self.distributed_models.keys():
            model, _ = self.distributed_models[block]
            utils.copy_model(self.center_model, model, self.device)

    def compute_deviation(self):
        # Synchronize the blocks with the central model
        for block_idx, block in enumerate(self.distributed_models.keys()):
            model, _ = self.distributed_models[block]
            utils.copy_block(model, self.center_model, block_idx)

        avg_param_diff_norm = self.average_parameter_difference_norm()
        self.param_deviation.append(avg_param_diff_norm)

        # Synchronize the central model with the blocks
        for block in self.distributed_models.keys():
            model, _ = self.distributed_models[block]
            utils.copy_model(self.center_model, model, self.device)
        

    def average_parameter_difference_norm(self):
        total_norm = 0

        # Iterate through each block's model
        for block, (model, _) in self.distributed_models.items():
            # Iterate through parameters of both the block's model and the central model
            for param_distributed, param_center in zip(model.parameters(), self.center_model.parameters()):
                # Compute the difference
                diff = param_distributed.data - param_center.data
                # Calculate the norm of the difference
                norm = torch.norm(diff)
                # Accumulate the total norm
                total_norm += norm.item()

        # Compute the average norm by dividing by the number of distributed models
        average_norm = total_norm / len(self.distributed_models) if self.distributed_models else 0

        return average_norm

class ImageClassifier_frame_entire:
    def __init__(self, model, H):
        self.H = H
        self.lr = H["step_size"]
        self.batch_size = H["batch_size"]
        self.rounds = H["rounds"]  # Number of training epochs
        self.device = torch.device(f"cuda:{H['cuda_core']}")
        self.dataset_name = H["dataset_name"]
        self.loss_history = []

        # Move the model to the specified device (e.g., GPU)
        self.model = model.to(self.device)

        # Initialize an optimizer for the entire model
        self.optimizers = [optim.Adam(self.model.parameters(), lr=self.lr)]

        # Define the loss function
        self.criterion = nn.CrossEntropyLoss()

        # Create a data loader for the training set
        self.train_loader = torch.utils.data.DataLoader(
            data.generate_imagedata(self.dataset_name),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        # Initialize the reporting mechanism
        self.reporter = utils.Reporter(H)

def get_num_classes(dataset_name):
    # Retrieve the dataset based on the given name
    dataset_raw = data.generate_imagedata(dataset_name)
    # Count the number of unique classes in the dataset
    num_classes = len(set([data[1] for data in dataset_raw]))
    # Release the dataset from memory
    del dataset_raw
    # Trigger garbage collection to free up memory
    gc.collect()
    
    return num_classes

def generate_ModelFrame(H):
    ttype = H["training_mode"]
    model_type = H.get("model", "ResNet18")  # Default to ResNet18 if not specified
    num_classes = get_num_classes(H["dataset_name"])
    pretrained = False

    # Initialize the model based on the specified type
    model = None
    if model_type == "ResNet18":
        model = Models.load_resnet18(pretrained=pretrained, num_classes=num_classes)
    elif model_type == "ResNet34":
        model = Models.load_resnet34(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Create the appropriate training frame based on the training mode
    if ttype == "entire":
        frame = ImageClassifier_frame_entire(model, H)
    elif ttype == "blockwise" or ttype == "blockwise_sequential":
        frame = ImageClassifier_frame_blockwise(model, H)
    else:
        raise ValueError(f"Unknown training type: {ttype}")

    # Set up signal handlers for graceful termination
    def signal_handler(sig, frame):
        if frame:
            frame.reporter.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    return frame