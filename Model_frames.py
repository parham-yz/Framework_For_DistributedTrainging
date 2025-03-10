import torch
import torch.nn as nn
import torch.optim as optim
import copy
import Models 
import utils
import signal
import sys

class Classifier_frame_blockwise:
    def __init__(self, H):
        self.H = H
        self.lr = H["step_size"]
        self.batch_size = H["batch_size"]
        self.rounds = H["rounds"]
        self.K = H["K"]
        self.device = torch.device(f"cuda:{H['cuda_core']}")
        self.dataset_name = H["dataset_name"]
        self.loss_history = []

        # Load dataset to determine the number of classes, then release the dataset from memory
        dataset_raw = utils.generate_data(self.dataset_name)
        num_classes = len(set([data[1] for data in dataset_raw]))
        del dataset_raw  # Remove the dataset from memory

        # Instantiate ResNet18 with the appropriate number of output heads
        self.center_model = Models.ResNet18(num_classes=num_classes).to(self.device)

        self.distributed_models = {}
        self.optimizers = []

        for i, block in enumerate(self.center_model.blocks):
            # Deep copy the model to create a separate model for each block
            copy_model = copy.deepcopy(self.center_model)
            # Create separate optimizers for each block using the copied model's parameters
            optimizer = optim.Adam(copy_model.blocks[i].parameters(), lr=self.lr)
            self.optimizers.append(optimizer)
            # Store the copied model and its optimizer in a dictionary
            self.distributed_models[f"block_{i}"] = (copy_model, optimizer)
            

        self.criterion = nn.CrossEntropyLoss()

        # Preload dataset to GPU and release the raw dataset from memory
        self.dataset = [(data[0].to(self.device), data[1]) for data in utils.generate_data(self.dataset_name)]
        # Split dataset into train and test sets (90% train, 10% test)
        train_size = int(1 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        self.big_train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size*8,
            shuffle=True,
            drop_last=True
        )

        # Initialize Reporter
        self.reporter = utils.Reporter(H)

    def communicate(self):
        for block_idx,block in enumerate(self.distributed_models.keys()):
            model ,_ = self.distributed_models[block]
            utils.copy_block(model,self.center_model, block_idx)

        for block in self.distributed_models.keys():
            model ,_ = self.distributed_models[block]
            utils.copy_model(self.center_model, model,self.device)

class Classifier_frame_entire:
    def __init__(self, H):
        self.H = H
        self.lr = H["step_size"]
        self.batch_size = H["batch_size"]
        self.rounds = H["rounds"]  # Use epochs for entire model training
        self.device = torch.device(f"cuda:{H['cuda_core']}" if torch.cuda.is_available() else "cpu")
        self.dataset_name = H["dataset_name"]
        self.loss_history = []

        # Load dataset to determine the number of classes, then release the dataset from memory
        dataset = utils.generate_data(self.dataset_name)
        num_classes = len(set([data[1] for data in dataset]))
        del dataset  # Remove the dataset from memory

        # Instantiate the model (e.g., ResNet18 with the appropriate number of output heads)
        self.model = Models.ResNet18(num_classes=num_classes).to(self.device)

        # Create a single optimizer for the entire model
        self.optimizers = [optim.Adam(self.model.parameters(), lr=self.lr)]

        self.criterion = nn.CrossEntropyLoss()

        # Preload dataset to GPU and release the raw dataset from memory
        self.train_loader = torch.utils.data.DataLoader(
            utils.generate_data(self.dataset_name),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        # Initialize Reporter
        self.reporter = utils.Reporter(H)


def generate_model(H):
    ttype = H["training_mode"]
    model = None
    if ttype == "entire":
        model = Classifier_frame_entire(H)
    elif ttype == "blockwise":
        model = Classifier_frame_blockwise(H)
    elif ttype == "blockwise_sequential":
        model = Classifier_frame_blockwise(H)
    else:
        raise ValueError(f"Unknown training type: {ttype}")

    # Register signal handlers to ensure reporters are terminated properly
    def signal_handler(sig, frame):
        if model:
            model.reporter.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    return model