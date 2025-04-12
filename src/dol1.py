import torch
import torch.nn as nn
import torch.optim as optim
import copy
import src.Optimizer_Engines.BCD_engine as BCD_engine
from src.Architectures.Models import load_resnet18, load_resnet34
from src.Data.data_imagenet import *
import utils
import src.Buliding_Units.Model_frames as Model_frames
import src.Buliding_Units.StopperUnit as StopperUnit


# Example usage (assuming H and other dependencies are defined)
# This script sets up the hyperparameters and model for training.
# It then selects the training mode based on the 'training_mode' parameter in H.
# Depending on the mode, it either trains blockwise or trains the entire model.

if __name__ == "__main__":
    H = {
        "step_size": 0.00001,
        "batch_size": 256,
        "rounds": 20000,
        "K": 1,
        "dataset_name": "cifar100",
        "cuda_core": -1,
        "training_mode": "blockwise_sequential",
        "report_sampling_rate" : 10,
        "model": "ResNet18",
        "communication_delay" : 1,
        "n_workers": 0.25
    }


    
    # Parse command-line arguments to update hyperparameters
    H = utils.parse_arguments(H)
    # Parse measurement units from the Units.txt file
    measurement_units = utils.parse_measurement_units()
    
    # Generate the model based on the specified training mode
    frame = Model_frames.generate_ModelFrame(H)
    frame.set_measure_units(measurement_units)
    frame.set_stopper_units([StopperUnit.AccuracyTargetStopper(0.9)])

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
    