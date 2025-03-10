import torch
import torch.nn as nn
import torch.optim as optim
import copy
import BCD_engine
from Models import load_resnet18, load_resnet34
from data_imagenet import *
import utils
import Model_frames



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
        "cuda_core": 0,
        "training_mode": "entire",
        "report_sampling_rate" : 10,
        "model": "ResNet34"
    }
    
    # Parse command-line arguments to update hyperparameters
    H = utils.parse_arguments(H)
    # Generate the model based on the specified training mode
    classifier = Model_frames.generate_ModelFrame(H)

    # Train the model using the specified training mode
    if H["training_mode"] == "blockwise":
        # Train the model blockwise if the training mode is 'blockwise'
        BCD_engine.train_blockwise(classifier)
    elif H["training_mode"] == "blockwise_sequential":
        # Train the entire model if the training mode is 'entire'
        BCD_engine.train_blockwise_sequential(classifier)
    elif H["training_mode"] == "entire":
        # Train the entire model if the training mode is 'entire'
        BCD_engine.train_entire(classifier)
    else:
        # Raise an error if an invalid training mode is specified
        raise ValueError(f"Invalid training mode: {H['training_mode']}")
    
    classifier.reporter.terminate()
    