import torch
import torch.nn as nn
import torch.optim as optim
import copy
import src.Optimizer_Engines.BCD_engine as BCD_engine
from src.Architectures.Models import load_resnet18, load_resnet34
import src.utils as utils
import src.Buliding_Units.Model_frames as Model_frames
import src.Buliding_Units.StopperUnit as StopperUnit
import time


# Example usage (assuming H and other dependencies are defined)
# This script sets up the hyperparameters and model for training.
# It then selects the training mode based on the 'training_mode' parameter in H.
# Depending on the mode, it either trains blockwise or trains the entire model.

H = {
    "step_size": None,
    "batch_size": None,
    "rounds": None,
    "dataset_name": None,
    "cuda_core": None,
    "training_mode": None,
    "report_sampling_rate": None,
    "measurement_sampling_rate": None,
    "model": None,
    "config": None,
    "K": None,
    "communication_delay": None,
    "n_workers": None,
    "reports_dir": None,
    "beta": None
}
H = {
    "step_size": 0.01,
    "batch_size": 256,
    "rounds": 3000,
    "dataset_name": "cifar10",
    "cuda_core": 0,
    "training_mode": "blockwise_sequential",
    "report_sampling_rate": 20,
    "measurement_sampling_rate": 400-1,
    "model": "ResNet34",
    "config": [32]*5,
    "beta":0.5,
    "K":1,
    "communication_delay":0,
    "n_workers":1,
    "reports_dir":"default_path"
}

# H = {
#     "step_size": 0.001,                 # --step_size
#     "batch_size": 128,                  # --batch_size
#     "rounds": 10000,                    # --rounds
#     "dataset_name": "cifar10",          # --dataset_name
#     "cuda_core": 0,                     # --cuda_core
#     "training_mode": "blockwise_sequential",  # --training_mode
#     "report_sampling_rate": 20,         # --report_sampling_rate
#     "measurement_sampling_rate": 30099,   # Default or computed as 400 - 1
#     "model": "residual_cnn",            # --model
#     "config": "[128, 64, 32,32,32,32,16,16,16,16,16,16,16,16,16,16,16,16]",             # --config
#     "K": 1,                             # --K
#     "communication_delay": 0,           # --communication_delay
#     "n_workers": 1,                     # Default value from the script
#     "reports_dir": "rcnn_cifar10"       # --reports_dir
# }

# H = {
#     "model": "residual_cnn",
#     "dataset_name": "cifar10",
#     "training_mode": "ploting",
#     "step_size": 1.1,
#     "batch_size": 128,
#     "rounds": 0,
#     "K": 0,
#     "cuda_core": 0,
#     "config": "[128, 64, 32,32,32,32,16,16,16,16,16,16,16,16,16,16,16,16]",
#     "communication_delay": 0,
#     "report_sampling_rate": 0,
#     "reports_dir": "model_arch"
# }



# Parse command-line arguments to update hyperparameters
H = utils.parse_arguments(H)
# Parse measurement units from the Units.txt file
measurement_units = utils.parse_measurement_units()

# Generate the model based on the specified training mode
_t0 = time.perf_counter()
frame = Model_frames.generate_ModelFrame(H)
frame.set_measure_units(measurement_units)
frame.set_stopper_units([StopperUnit.AccuracyTargetStopper(0.9)])
init_elapsed = time.perf_counter() - _t0
frame.reporter.log(f"Frame initialization and attribute setup took {init_elapsed:.2f} seconds")
# Report total number of parameters in the model
total_params = sum(p.numel() for p in frame.center_model.parameters())
frame.reporter.log(f"Total number of parameters: {total_params:,}")
# Train the model using the specified training mode
if H["training_mode"] == "blockwise":
    # Train the model blockwise if the training mode is 'blockwise'
    BCD_engine.train_blockwise_distributed(frame)
elif H["training_mode"] == "blockwise_sequential":
    # Train the entire model if the training mode is 'entire'
    BCD_engine.train_blockwise_sequential(frame)
elif H["training_mode"] == "entire":
    # Train the entire model if the training mode is 'entire'
    BCD_engine.train_entire(frame)
elif H["training_mode"] == "ploting":
    # Train the entire model if the training mode is 'entire'
    utils.export_pytorch_to_onnx(frame.center_model, frame.input_shape, "reports/"+H["reports_dir"] + "/model.onnx")
else:
    # Raise an error if an invalid training mode is specified
    raise ValueError(f"Invalid training mode: {H['training_mode']}")

frame.reporter.terminate()
