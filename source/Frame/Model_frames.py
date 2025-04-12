import torch
import torch.nn as nn
import torch.optim as optim
import copy
import FramWork_For_DistributedNNTrainging.source.Architectures.Models as Models 
import utils
import signal
import sys
import data
import gc
import time


class Frame:
    def __init__(self, model, H):
        self.H = H
        self.lr = H["step_size"]
        self.batch_size = H["batch_size"]
        if H["cuda_core"] == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{H['cuda_core']}")
        self.center_model = model.to(self.device)
        self.reporter = utils.Reporter(H)
        self.criterion = nn.CrossEntropyLoss()  # Common loss function
        
        # Initialize the list of measurement units (instances of MeasurementUnit subclasses)
        self.measure_units = []
        # Initialize the list of stopper units (instances of StopperUnit subclasses)
        self.stopper_units = []

    def setup_dataloaders(self, data_set):
        self.dataset = [(d[0].to(self.device), d[1]) for d in data_set]
        train_size = int(0.9 * len(self.dataset))
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
            batch_size=self.batch_size * 8,
            shuffle=True,
            drop_last=True
        )

        
    def set_measure_units(self, measure_units: list):
        """
        Set the measurement units to be used for logging metrics.
        
        Args:
            measure_units (list): List of MeasurementUnit instances.
        """
        self.measure_units = measure_units
    
    def set_stopper_units(self, stopper_units: list):
        """
        Set the stopper_units units.
        
        Args:
            stopper_units (list): List of StopperUnit instances.
        """
        self.stopper_units = stopper_units

    def run_measurmentUnits(self):
        """
        Runs each measurement unit by feeding the frame.
        """
        
        # Feed the models list to each measurement unit
        for measurement_unit in self.measure_units:
            result = measurement_unit.measure(self)
            measurement_unit.log_measurement(result)


class Disributed_frame(Frame):
    def __init__(self, model, H):
        super().__init__(model, H)

        self.K = H['K']

        self.distributed_models = {}
        self.optimizers = []


    def train(self):
        """
        Abstract method for distributed training.
        Subclasses should implement this method to define their distributed training logic.
        """
        raise NotImplementedError("Subclasses must implement the train() method for Disributed_frame.")

    def communicate_withDelay(self):
        # Synchronize the blocks with the central model
        for block_idx, block in enumerate(self.distributed_models.keys()):
            model, _ = self.distributed_models[block]
            utils.copy_block(model, self.center_model, block_idx)

        time.sleep(self.H['communication_delay'])
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

    def init_distributed_models(self):
        # If the model has a block structure, create a distributed copy for each block.
        if hasattr(self.center_model, "blocks"):
            for i, block in enumerate(self.center_model.blocks):
                copy_model = copy.deepcopy(self.center_model)
                optimizer = optim.Adam(copy_model.blocks[i].parameters(), lr=self.lr)
                self.optimizers.append(optimizer)
                self.distributed_models[f"block_{i}"] = (copy_model, optimizer)
        else:
            raise ValueError("Distributed regression training requires a block-structured model. The provided center_model does not have a 'blocks' attribute.")

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


class ImageClassifier_frame_blockwise(Disributed_frame):
    def __init__(self, model, H):
        super().__init__(model, H)
        self.rounds = H["rounds"]
        self.K = H["K"]
        self.dataset_name = H["dataset_name"]
        self.loss_history = []
        self.param_deviation = []


        # # Iterate over the blocks of the model
        # for i, block in enumerate(self.center_model.blocks):
        #     # Create a deep copy of the model for each block
        #     copy_model = copy.deepcopy(self.center_model)
        #     # Initialize an optimizer for the parameters of the current block
        #     optimizer = optim.Adam(copy_model.blocks[i].parameters(), lr=self.lr)
        #     self.optimizers.append(optimizer)
        #     # Map each block to its corresponding model and optimizer
        #     self.distributed_models[f"block_{i}"] = (copy_model, optimizer)


        self.init_distributed_models()

        # Define the loss function
        self.criterion = nn.CrossEntropyLoss()

        # Load the dataset into memory and transfer it to the device
        dataset= data.generate_imagedata(self.dataset_name)
        self.setup_dataloaders(dataset)

        # Log additional information about frame type, number of blocks, and number of model parameters
        frame_type = type(self).__name__
        num_blocks = len(self.center_model.blocks)
        num_parameters = sum(p.numel() for p in self.center_model.parameters())
        self.reporter.log(f"Frame Type: {frame_type}, Number of Blocks: {num_blocks}, Number of Model Parameters: {num_parameters}")


class Regression_frame_blockwis(Disributed_frame):
    def __init__(self, model, H):
        super().__init__(model, H)
        self.rounds = H["rounds"]

        self.dataset_name = H["dataset_name"]
        self.loss_history = []
        self.param_deviation = []
        


        self.init_distributed_models()

        # Use Mean Squared Error Loss for regression tasks.
        self.criterion = nn.MSELoss()

        # Attempt to load a regression dataset. If data.generate_regressiondata is not defined,
        dataset= data.generate_regressiondata(self.dataset_name)
        self.setup_dataloaders(dataset)

        # Log additional information about frame type, number of blocks, and number of model parameters
        frame_type = type(self).__name__
        num_blocks = len(self.distributed_models)
        num_parameters = sum(p.numel() for p in self.center_model.parameters())
        self.reporter.log(f"Frame Type: {frame_type}, Number of Blocks: {num_blocks}, Number of Model Parameters: {num_parameters}")


class ImageClassifier_frame_entire(Frame):
    def __init__(self, model, H):
        super().__init__(model, H)
        self.rounds = H["rounds"]  # Number of training epochs
        self.dataset_name = H["dataset_name"]
        self.loss_history = []



        # Initialize an optimizer for the entire model
        self.optimizers = [optim.Adam(self.center_model.parameters(), lr=self.lr)]

        # Define the loss function
        self.criterion = nn.CrossEntropyLoss()

        # Create a data loader for the training set
        dataset= data.generate_regressiondata(self.dataset_name)
        self.setup_dataloaders(dataset)


def get_dataset_dimensionality(dataset_name,dataset_type):
    # Retrieve the dataset based on the given name
    if dataset_type =='image':
        dataset_raw = data.generate_imagedata(dataset_name)
    elif dataset_type == 'nlp':
        dataset_raw = data.generate_nlp_data(dataset_name)
    elif dataset_type == 'regression':
        dataset_raw = data.generate_regressiondata(dataset_name)

    # Determine the input and target shapes of the dataset
    input_shape = dataset_raw[0][0].shape
    target_shape = dataset_raw[0][1].shape if hasattr(dataset_raw[0][1], 'shape') else (1,)
    # Release the dataset from memory
    del dataset_raw
    # Trigger garbage collection to free up memory
    gc.collect()
    
    return input_shape, target_shape

# The following function, `generate_ModelFrame`, is responsible for creating and initializing
# a model frame based on the hyperparameters and model type specified in the dictionary `H`.
# It supports different types of models, including image models (e.g., ResNet18, ResNet34),
# regression models (e.g., linear_nn), and NLP models. The function first determines the
# input and output shapes of the dataset, then initializes the model accordingly. Finally,
# it creates the appropriate training frame based on the training mode specified in `H`.

def generate_ModelFrame(H):
    image_models = ["ResNet18","ResNet34"]
    regression_models = ["linear_nn"]
    nlp_models = [""]

    


    ttype = H["training_mode"]
    model_type = H.get("model", "ResNet18")  # Default to ResNet18 if not specified
    
    if model_type in image_models:
        input_shape, output_shape = get_dataset_dimensionality(H["dataset_name"],'image')
    elif model_type in regression_models:
        input_shape, output_shape = get_dataset_dimensionality(H["dataset_name"],'regression')
    elif model_type in nlp_models:
        input_shape, output_shape = get_dataset_dimensionality(H["dataset_name"],'nlp')

    pretrained = False

    # Initialize the model based on the specified type
    model = None
    if model_type == "ResNet18":
        model = Models.load_resnet18(pretrained=pretrained, num_classes=output_shape)
    elif model_type == "ResNet34":
        model = Models.load_resnet34(pretrained=pretrained, num_classes=output_shape)
    elif model_type == "linear_nn":
        # Load a feedforward network for regression tasks; use defaults if not provided in H.
        config = [32]*16
        assert len(input_shape) == 1, "Expected input_shape to have length 1 for linear_nn"
        assert len(output_shape) == 1, "Expected output_shape to have length 1 for linear_nn"
        input_dim = input_shape[0]
        output_dim = output_shape[0]
        activation = nn.Identity()
        final_activation = nn.Identity()
        model = Models.load_feedforward(config, input_dim, output_dim, activation, final_activation)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    #

    # Create the appropriate training frame
    if ttype == "entire":
        frame = ImageClassifier_frame_entire(model, H)

    elif ttype == "blockwise" or ttype == "blockwise_sequential":

        if model_type in regression_models:

            frame = Regression_frame_blockwis(model, H)

        elif model_type in image_models:

            frame = ImageClassifier_frame_blockwise(model, H)
    else:
        raise ValueError(f"Unknown training type: {ttype}")

    # Set up signal handlers for graceful termination
    def signal_handler(sig, frame_obj):
        if frame:
            frame.reporter.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    return frame