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

if __name__ == "__main__":
    H = {
        "step_size": 0.01,
        "batch_size": 256,
        "rounds": 3000,
        "dataset_name": "mini_mnist_8chanel",
        "cuda_core": 0,
        "training_mode": "entire",
        "report_sampling_rate": 20,
        "measurement_sampling_rate": 400-1,
        "model": "residual_cnn",
        "config": [8]*5,
        "beta":0.5
    }
    H = {
        "step_size": 0.01,
        "batch_size": 256,
        "rounds": 3000,
        "dataset_name": "mini_imagenet",
        "cuda_core": 0,
        "training_mode": "entire",
        "report_sampling_rate": 20,
        "measurement_sampling_rate": 400-1,
        "model": "residual_cnn",
        "config": [8]*5,
        "beta":0.5,
        "K":1,
        "communication_delay":0,
        "n_workers":1,
        "reports_dir":"default_path"
    }


    
    # Parse command-line arguments to update hyperparameters
    H = utils.parse_arguments(H)
    # Parse measurement units from the Units.txt file
    measurement_units = utils.parse_measurement_units()
    
    # Generate the model based on the specified training mode
    _t0 = time.perf_counter()
    frame = Model_frames.generate_ModelFrame(H)
    frame.set_measure_units(measurement_units)
    frame.set_stopper_units([StopperUnit.AccuracyTargetStopper(0.95)])
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
    else:
        # Raise an error if an invalid training mode is specified
        raise ValueError(f"Invalid training mode: {H['training_mode']}")
    
    frame.reporter.terminate()
    