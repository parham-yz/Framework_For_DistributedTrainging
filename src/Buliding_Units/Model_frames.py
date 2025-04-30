import torch
import torch.nn as nn
import torch.optim as optim
import copy
import src.Architectures.Models as Models 
import src.utils as utils
import signal
import sys
import src.Data.data_sets as data_sets
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

        self.input_shape  = None
        self.output_shape = None     

    def setup_dataloaders(self, data_set,layzzy_loader=False):
        # Keep the original dataset structure, avoid moving everything to device here
        self.dataset = data_set

        train_size = int(0.9 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

        # Determine the number of workers. A common heuristic is the number of CPU cores.
        # You might need to experiment to find the optimal number for your system and data.

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        

        # Assuming big_train_loader is for a specific purpose that requires a larger batch
        self.big_train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size * 8,
            shuffle=True,
            drop_last=True
        )


        # For the test loader, shuffling is typically not necessary
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size, # Or whatever batch size is appropriate for testing
            shuffle=False,
            drop_last=False # Usually don't drop the last batch in testing
        )

        if not layzzy_loader:
            train_data = [(x[0].to(self.device, non_blocking=False),x[1].to(self.device, non_blocking=False)) for x in self.train_loader]
            train_data_bigBatch = [(x[0].to(self.device, non_blocking=False),x[1].to(self.device, non_blocking=False)) for x in self.big_train_loader]
            # test_data = [(x[0].to(self.device, non_blocking=False),x[1].to(self.device, non_blocking=False)) for x in self.test_loader]
            
            self.train_loader = train_data
            self.big_train_loader = train_data_bigBatch
            # self.test_loader = test_data

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
        # Log start of all measurement units
        import time as _time
        self.reporter.log("Starting measurement units")
        _t0 = _time.perf_counter()
        # Run each measurement unit by feeding the frame and reporting results
        for measurement_unit in self.measure_units:
            unit_name = measurement_unit.__class__.__name__
            # Log before measurement
            self.reporter.log(f"Starting measurement: {unit_name}")
            # Perform measurement
            result = measurement_unit.measure(self)
            # Log to the unit's own measurement log
            measurement_unit.log_measurement(result)
        # Log completion of all measurement units with timing
        elapsed = _time.perf_counter() - _t0
        self.reporter.log(f"Finished measurement units , took {elapsed:.2f} seconds")


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
                copy_model = copy.deepcopy(self.center_model).to(self.device)
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
        dataset= data_sets.generate_imagedata(self.dataset_name)
        self.setup_dataloaders(dataset)

        # Log additional information about frame type, number of blocks, and number of model parameters
        frame_type = type(self).__name__
        num_blocks = len(self.center_model.blocks)
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
        dataset= data_sets.generate_imagedata(self.dataset_name)
        self.setup_dataloaders(dataset)


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
        dataset= data_sets.generate_regressiondata(self.dataset_name)
        self.setup_dataloaders(dataset)

        # Log additional information about frame type, number of blocks, and number of model parameters
        frame_type = type(self).__name__
        num_blocks = len(self.distributed_models)
        num_parameters = sum(p.numel() for p in self.center_model.parameters())
        self.reporter.log(f"Frame Type: {frame_type}, Number of Blocks: {num_blocks}, Number of Model Parameters: {num_parameters}")

class Regression_frame_entire(Frame):
    """
    A training frame for regression tasks that trains the entire model at once.

    Inherits from the base Frame class and sets up components specific to
    entire-model regression training, such as MSELoss criterion and loading
    regression datasets.
    """
    def __init__(self, model, H):
        """
        Initializes the Regression_frame_entire.

        Args:
            model (nn.Module): The regression model to be trained.
            H (dict): Dictionary of hyperparameters. Expected keys include:
                      "rounds", "dataset_name", "step_size", "batch_size",
                      "cuda_core".
        """
        super().__init__(model, H)
        self.rounds = H["rounds"]  # Number of training epochs/rounds
        self.dataset_name = H["dataset_name"]
        self.loss_history = [] # To store training loss values

        # Initialize an optimizer for the entire model's parameters
        # Stored in a list for potential consistency with distributed frames,
        # though only one optimizer is used here.
        self.optimizers = [optim.Adam(self.center_model.parameters(), lr=self.lr)]

        # Define the loss function suitable for regression
        self.criterion = nn.MSELoss() # Mean Squared Error Loss

        # Load the appropriate regression dataset
        try:
            dataset = data_sets.generate_regressiondata(self.dataset_name)
        except ValueError as e:
            self.reporter.log(f"Error loading regression dataset '{self.dataset_name}': {e}")
            raise # Re-raise the error after logging
        except Exception as e:
            self.reporter.log(f"An unexpected error occurred during dataset loading: {e}")
            raise

        # Setup the dataloaders for training and testing
        # The setup_dataloaders method splits data and creates loaders
        self.setup_dataloaders(dataset) # Use layzzy_loader=False by default if desired

        # Log basic information about the frame and model
        frame_type = type(self).__name__
        num_parameters = sum(p.numel() for p in self.center_model.parameters() if p.requires_grad)
        self.reporter.log(f"Frame Type: {frame_type}, Trainable Parameters: {num_parameters}")
        # Optionally log input/output shapes if determined (usually set later in generate_ModelFrame)
        # if self.input_shape and self.output_shape:
        #    self.reporter.log(f"Expected Input Shape (batched): {self.input_shape}")
        #    self.reporter.log(f"Expected Output Shape (batched): {self.output_shape}")


# --- Modification in generate_ModelFrame ---
# You need to update the `generate_ModelFrame` function to instantiate
# this new class when appropriate.



def get_dataset_dimensionality(dataset_name,dataset_type):
    # Retrieve the dataset based on the given name
    if dataset_type =='image':
        dataset = data_sets.generate_imagedata(dataset_name)
    elif dataset_type == 'nlp':
        dataset = data_sets.generate_nlp_data(dataset_name)
    elif dataset_type == 'regression':
        dataset = data_sets.generate_regressiondata(dataset_name)

    # Determine the *transformed* input shape rather than raw stored data.
    try:
        # Most torchvision datasets return (image, label) tuples and apply the
        # defined transform inside __getitem__. Fetch a single sample to inspect
        # the true tensor shape that the model will receive.
        sample_input = dataset[0][0]
        # If the transform returns PIL image instead of tensor, convert to tensor
        if hasattr(sample_input, "shape"):
            input_shape = sample_input.shape
        else:
            # Fallback: use raw data attribute as before.
            input_shape = dataset.data[0].shape
    except Exception:
        # Fallback for datasets that don’t support indexing like this.
        input_shape = dataset.data[0].shape

    dataset.targets = torch.tensor(dataset.targets)
    if dataset.targets.dim() == 1:
        target_shape = (1,)
    else:
        target_shape = dataset.targets[0].shape

    if target_shape[0] == 1 and dataset_type == 'image':
        # It's a classification task, count the number of classes
        num_classes = max(dataset.targets) + 1
        target_shape = (num_classes,)
        
    # Release the dataset from memory
    del dataset
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

    from src.Data.data_sets import IMAGE_DATASETS, REGRESSION_DATASETS, NLP_DATASETS

    ttype = H["training_mode"]
    model_type = H.get("model", "ResNet18")  # Default to ResNet18 if not specified
    dataset_name = H["dataset_name"]

    if dataset_name in IMAGE_DATASETS:
        input_shape, output_shape = get_dataset_dimensionality(dataset_name, 'image')
    elif dataset_name in REGRESSION_DATASETS:
        input_shape, output_shape = get_dataset_dimensionality(dataset_name, 'regression')
    elif dataset_name in NLP_DATASETS:
        input_shape, output_shape = get_dataset_dimensionality(dataset_name, 'nlp')
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    pretrained = False

    # Initialize the model based on the specified type
    model = None
    if model_type == "ResNet18":
        model = Models.load_resnet18(pretrained=pretrained, num_classes=output_shape[0])

    elif model_type == "ResNet34":
        print(f"\n\n\nin_features: {input_shape}, num_classes: {output_shape}\n\n\n")
        model = Models.load_resnet34(pretrained=pretrained, num_classes=output_shape[0], bi_partitioned=False)

    elif model_type == "ResNet34-bi":
        print(f"\n\n\nin_features: {input_shape}, num_classes: {output_shape}\n\n\n")
        model = Models.load_resnet34(pretrained=pretrained, num_classes=output_shape[0], bi_partitioned=True)

    elif model_type == "linear_nn":
        # Load a feedforward network for regression tasks; use defaults if not provided in H.
        config = H["config"]
        assert len(input_shape) == 1, "Expected input_shape to have length 1 for linear_nn"
        assert len(output_shape) == 1, "Expected output_shape to have length 1 for linear_nn"
        input_dim = input_shape[0]
        output_dim = output_shape[0]
        activation = nn.Identity()
        final_activation = nn.Identity()
        model = Models.load_feedforward(config, input_dim, output_dim, activation, final_activation)

    elif model_type == "neural_net":
        # Load a feedforward network for regression tasks; use defaults if not provided in H.
        config = H["config"]
        assert len(input_shape) == 1, "Expected input_shape to have length 1 for linear_nn"
        assert len(output_shape) == 1, "Expected output_shape to have length 1 for linear_nn"
        input_dim = input_shape[0]
        output_dim = output_shape[0]
        activation = nn.ReLU()
        final_activation = nn.Identity()
        model = Models.load_feedforward(config, input_dim, output_dim, activation, final_activation)

    elif model_type == "cnn":
        # Load a feedforward network for regression tasks; use defaults if not provided in H.
        config = H["config"]
        assert len(output_shape) == 1, "Expected output_shape to have length 1 for cnn"
        output_dim = output_shape[0]
        input_dim = input_shape[0]
        activation = nn.ReLU()
        final_activation = None
        model = Models.load_feedforward_cnn(config,input_dim, output_dim, activation, final_activation)

    # ------------------------------------------------------------------
    # Residual Feed‑Forward CNN
    # ------------------------------------------------------------------
    elif model_type in ["residual_cnn"]:
        config = H["config"]
        assert len(output_shape) == 1, "Expected output_shape to have length 1 for residual_cnn"
        output_dim = output_shape[0]
        activation = nn.ReLU()
        final_activation = None

        in_channels = input_shape[0] if len(input_shape) > 0 else 3

        model = Models.load_residual_feedforward_cnn(
            config,
            in_channels,
            output_dim,
            activation,
            final_activation,
            bi_partitioned = False
        )
    
    elif model_type in ["residual_cnn_bi"]:
        config = H["config"]
        assert len(output_shape) == 1, "Expected output_shape to have length 1 for residual_cnn"
        output_dim = output_shape[0]
        activation = nn.ReLU()
        final_activation = None

        in_channels = input_shape[0] if len(input_shape) > 0 else 3

        model = Models.load_residual_feedforward_cnn(
            config,
            in_channels,
            output_dim,
            activation,
            final_activation,
            bi_partitioned = True
        )

    # ------------------------------------------------------------------
    # Ensemble of feed‑forward CNNs
    # ------------------------------------------------------------------
    elif model_type in ["cnn_ensemble", "feedforward_cnn_ensemble"]:
        configs = H["config"]  # 2‑D list of conv channels per sub‑model
        assert len(output_shape) == 1, "Expected output_shape to have length 1 for CNN ensemble"
        output_dim = output_shape[0]

        in_channels = input_shape[0] if len(input_shape) > 0 else 3  # fallback to 3
        activation = nn.ReLU()
        final_activation = None

        model = Models.load_feedforward_cnn_ensemble(
            configs, in_channels, output_dim, activation, final_activation
        )

    # ------------------------------------------------------------------
    # Ensemble of feed‑forward networks (vector input)
    # ------------------------------------------------------------------
    elif model_type in [ "feedforward_ensemble"]:
        configs = H["config"]  # 2‑D list
        assert len(input_shape) == 1, "Expected input_shape to have length 1 for ensemble feed‑forward model"
        assert len(output_shape) == 1, "Expected output_shape to have length 1 for ensemble feed‑forward model"

        input_dim = input_shape[0]
        output_dim = output_shape[0]

        # Keep same activation choices as linear_nn (completely linear by default). Users can
        # still override by passing a different 'activation' in *H* later if desired.
        activation = nn.ReLU()
        final_activation = nn.Identity()

        model = Models.load_feedforward_ensemble(
            configs, input_dim, output_dim, activation, final_activation
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    #

    # Create the appropriate training frame
    if ttype == "entire" or ttype == "ploting": # Changed 'ploting' to 'plotting' assuming typo
        # Select frame based on dataset type for 'entire' mode
        if dataset_name in REGRESSION_DATASETS:
            frame = Regression_frame_entire(model, H)      # Use the new class
        elif dataset_name in IMAGE_DATASETS:
            frame = ImageClassifier_frame_entire(model, H) # Existing class
        # elif dataset_type == 'nlp':
        #     # frame = NLP_frame_entire(model, H) # If you create this
        #     raise NotImplementedError("Entire training mode not yet implemented for NLP datasets.")
        else:
             raise ValueError(f"Unsupported dataset type '{data_sets}' for 'entire' training mode.")

    elif ttype == "blockwise" or ttype == "blockwise_sequential":
        # Select appropriate distributed training frame based on dataset type
        if dataset_name in REGRESSION_DATASETS:
            frame = Regression_frame_blockwis(model, H)
        elif dataset_name in IMAGE_DATASETS:
            frame = ImageClassifier_frame_blockwise(model, H)
        # elif dataset_type == 'nlp':
        #     # frame = NLP_frame_blockwise(model, H) # If you create this
        #     raise NotImplementedError("Blockwise training mode not yet implemented for NLP datasets.")
        else:
            raise ValueError(f"Unsupported dataset type '{data_sets}' for 'blockwise' training mode.")
    else:
        raise ValueError(f"Unknown training type: {ttype}")

    # Set up signal handlers for graceful termination
    def signal_handler(sig, frame_obj):
        if frame:
            frame.reporter.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    frame.input_shape = torch.Size([H["batch_size"], *input_shape])
    frame.output_shape = torch.Size([H["batch_size"], *output_shape])

    return frame