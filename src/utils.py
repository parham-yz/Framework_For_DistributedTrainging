import logging
try:
    import GPUtil
    import inspect
    import src.Buliding_Units.MeasurementUnit as MeasurementUnit
    
except ImportError:
    import subprocess
    import sys
    logging.info("GPUtil not found. Installing GPUtil...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "GPUtil"])
    import GPUtil


# Ensure onnx is installed
try:
    import onnx
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx"])
    import onnx


import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import subprocess
import hashlib
import os
import json
import datetime
from torchvision.datasets import ImageFolder
import argparse
from tqdm import tqdm
import copy
import ast
import os
import  src.Buliding_Units.MeasurementUnit as MeasurementUnit
import torch.onnx
from torch.onnx import TrainingMode

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
            transforms.Grayscale(num_output_channels=3),
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

class Reporter:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        hyperparameters_str = json.dumps(hyperparameters, sort_keys=True)
        hash_object = hashlib.md5(hyperparameters_str.encode())
        hash_hex = hash_object.hexdigest()[:8]
        
        reports_subdir = hyperparameters.get("reports_dir", "default_path")
        reports_path = os.path.join("reports", reports_subdir)
        self.log_filename = os.path.join(reports_path, f"R{hash_hex}.txt")
        self.reports_path = reports_path
        self.ensure_log_file()

    def ensure_log_file(self):
        # Create the reports directory if it doesn't exist
        if not os.path.exists(self.reports_path):
            os.makedirs(self.reports_path)
        
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



def copy_block(source_model, target_model, block_index):
    """
    Copies the parameters of a specific block from the source model to the target model.
    
    Args:
        source_model: The model from which to copy the parameters.
        target_model: The model to which the parameters will be copied.
        block_index: The index of the block whose parameters are to be copied.
    """
    # Ensure the block index is within the range of available blocks
    if block_index < 0 or block_index >= len(source_model.blocks):
        raise IndexError("Block index is out of range.")
    
    # Get the specific block from both models
    source_block = source_model.blocks[block_index]
    target_block = target_model.blocks[block_index]
    
    # Iterate over the parameters of the source block
    with torch.no_grad():  # Ensure no gradients are computed during parameter copying
        for source_param, target_param in zip(source_block.parameters(), target_block.parameters()):
            # Copy the parameter data to the target model's block
            # .data ensures we are modifying the tensor in-place and detach() ensures no gradient history is copied
            target_param.data = copy.deepcopy(source_param.data).detach()


def parse_arguments(H):
    """
    Dynamically parses command-line arguments based on the fields of an input dictionary H,
    and updates H with any new values provided via the command line.

    If a command-line argument's value is a string in the format "[...]"
    (e.g., "[1, 'hello', 3.14]"), it attempts to parse it into a Python list.
    Otherwise, the value retains the type determined by argparse or remains a string.

    Args:
        H (dict): Initial dictionary containing default values for arguments.

    Returns:
        H (dict): Updated dictionary containing the parsed command-line arguments,
                  with list conversions applied where applicable.
    """
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Dynamically process input arguments for training configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )

    # Dynamically add arguments based on the keys and values of H
    for key, value in H.items():
        # Determine the type of the default value
        arg_type = type(value)

        if arg_type is bool:
            # Special handling for boolean flags: allows --key True or --key False
            # Note: argparse by default treats '--key' as setting True (action='store_true').
            # Using a type lambda allows explicit True/False.
            parser.add_argument(
                f"--{key}",
                type=lambda x: (str(x).lower() == 'true'),
                default=value,
                help=f"Boolean flag for {key}"
            )
        # Handle list defaults - expect string input like "[1,2]" for override
        elif arg_type is list:
             parser.add_argument(
                f"--{key}",
                type=str, # Expect a string input from command line
                default=str(value), # Represent default as string in help message
                help=f"{key} (provide as a string like '[item1, item2,...]')"
             )
        else:
            # Handles int, float, str, etc.
            parser.add_argument(
                f"--{key}",
                type=arg_type,
                default=value,
                help=f"{key} (type: {arg_type.__name__})"
            )

    # Parse arguments from sys.argv (command line)
    args = parser.parse_args()

    # Update H with arguments provided via command line, applying list conversion
    parsed_args_dict = vars(args)
    for key, value in parsed_args_dict.items():
        # Check if the key exists in the original H (to avoid processing unrelated args if any)
        # and if the parsed value is a string that looks like a list literal
        if key in H and isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            try:
                # Attempt to safely evaluate the string as a Python literal
                parsed_value = ast.literal_eval(value)
                # Check if the result is actually a list
                if isinstance(parsed_value, list):
                    H[key] = parsed_value
                else:
                    # It evaluated, but not to a list (e.g., user provided "[1]")
                    # Keep the original parsed value (which was the string)
                    H[key] = value
                    print(f"Warning: Argument --{key} value '{value}' evaluated but is not a list type. Keeping as string.", file=sys.stderr)
            except (ValueError, SyntaxError, TypeError) as e:
                # Evaluation failed (invalid format inside brackets)
                # Keep the original string value and issue a warning
                H[key] = value
                print(f"Warning: Could not parse argument --{key} value '{value}' as a list due to error: {e}. Keeping as string.", file=sys.stderr)
        else:
            # Otherwise, update H with the parsed value directly
            # This handles booleans, ints, floats, and strings not matching the list format
            # It also handles the case where the default was a list, but the user didn't override it
            # In that case, `value` would still be the default list object (if argparse didn't change it based on type=str).
            # Let's refine: if the default was list and the value is the string representation of the default, convert it back.
            # Or better: If the arg type in H was list and the current value is string, try conversion.
            if key in H and isinstance(H[key], list) and isinstance(value, str):
                 # This case handles when the default was a list, argparse stored the default as string (because type=str),
                 # and the user didn't provide the argument. We should restore the original list default.
                 # Let's compare with the string representation of the original default.
                 if value == str(H[key]): # Check if the string value is just the stringified default
                      H[key] = H[key] # Restore the original list default
                 else:
                     # User provided a string, but it didn't match the [...] format handled above. Keep as string.
                     H[key] = value
            else:
                 # Default update for all other types or values already correctly typed.
                 H[key] = value


    return H


def copy_model(source_model, target_model,device):
    """
    Copies the parameters from source_model to target_model.

    Parameters:
    - source_model: A PyTorch model from which to copy the parameters.
    - target_model: A PyTorch model to which the parameters will be copied.

    Both models should be of the same architecture.
    """
    # Ensure that both models are on the same device to avoid unnecessary data transfers
    target_model.to(device)
    # Get the state dictionary of the source model
    source_state_dict = source_model.state_dict()
    # Load the source state dictionary into the target model
    target_model.load_state_dict(source_state_dict)



def parse_measurement_units():
    """
    Parses the measurement units from the Units.txt file, instantiates the corresponding classes,
    and returns a list with the measurement unit instances.

    The expected file format is:
        Measurement Units In Use:
        - Working_memory_usage

    You can extend the unit_map below to support additional measurement unit classes.
    """
    # Set default file path to Units.txt in the same directory as this file.
    
    file_path = os.path.join("measurements", "Units.txt")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Units file not found: {file_path}")

    # Map valid unit names (as they appear in the file) to the corresponding classes.


    # Dynamically find all classes in MeasurementUnit that are subclasses of MeasurementUnit
    unit_map = {
        cls.__name__: cls for _, cls in inspect.getmembers(MeasurementUnit, inspect.isclass)
        if issubclass(cls, MeasurementUnit.MeasurementUnit) and cls is not MeasurementUnit.MeasurementUnit
    }
    measurement_units = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Loop through each line and check for lines that describe a measurement unit.
    for line in lines:
        stripped = line.strip()
        # Look for lines that start with a dash
        if stripped.startswith("-"):
            unit_name = stripped[1:].strip()
            if unit_name in unit_map:
                # Instantiate the measurement unit and add it to the list.
                measurement_units.append(unit_map[unit_name]())
            else:
                print(unit_map)
                raise ValueError(f"Unknown measurement unit: {unit_name}")
                
    return measurement_units




def export_pytorch_to_onnx(model, input_shape, output_path, input_names=None, output_names=None, dynamic_axes=None, verbose=False):
    """
    Exports a PyTorch model to the ONNX format, ensuring operations run on the CPU.

    Args:
        model (torch.nn.Module): The PyTorch model to export. It will be moved to CPU internally.
        input_shape (tuple or list): The shape of the input tensor the model expects,
                                     including the batch size dimension.
                                     Example: (1, 3, 224, 224) for a single image.
        output_path (str): The file path where the ONNX model will be saved (e.g., 'model.onnx').
        input_names (list[str], optional): Names to assign to the input nodes in the ONNX graph.
                                           Defaults to ['input'].
        output_names (list[str], optional): Names to assign to the output nodes in the ONNX graph.
                                            Defaults to ['output'].
        dynamic_axes (dict, optional): Specifies dynamic axes for inputs/outputs.
                                        Useful if your model accepts inputs of variable sizes (e.g., batch size).
                                        Example: {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                                        Defaults to None.
        verbose (bool, optional): If True, prints detailed information during the export process.
                                  Defaults to False.

    Returns:
        None: The function saves the ONNX model to the specified `output_path`.

    Raises:
        Exception: Catches and prints potential errors during the export process.
        ValueError: If input_shape is not provided or is invalid.
    """
    if not input_shape or not isinstance(input_shape, (tuple, list)):
        raise ValueError("`input_shape` must be provided as a tuple or list (e.g., (1, 3, 224, 224)).")

    try:
        # --- CPU Handling ---
        # Define the device to use (CPU)
        device = torch.device("cpu")
        # Move the model to the specified device (CPU)
        model.to(device)
        # Ensure the model is in evaluation mode (important for layers like Dropout, BatchNorm)
        model.eval()
        print(f"Model moved to {device} and set to evaluation mode.")
        # --------------------

        # Create the dummy input tensor directly on the CPU
        # Note: This assumes the input type is float. Adjust if your model needs a different dtype (e.g., torch.long).
        dummy_input = torch.randn(input_shape, requires_grad=False, device=device)
        print(f"Created dummy input with shape: {dummy_input.shape} on device: {dummy_input.device}")

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")

        print(f"Starting ONNX export to: {output_path}")

        # Export the model
        torch.onnx.export(
            model,                    # model being run (already on CPU)
            dummy_input,              # model input (already on CPU)
            output_path,              # where to save the model (can be file or file-like object)
            export_params=True,       # store the trained parameter weights inside the model file
            opset_version=11,         # the ONNX version to export the model to (choose appropriate version)
            do_constant_folding=True, # whether to execute constant folding for optimization
            input_names=input_names or ['input'],     # the model's input names
            output_names=output_names or ['output'],  # the model's output names
            dynamic_axes=dynamic_axes, # dynamic axes for variable length input/output
            verbose=verbose,           # print verbose logs
            training=TrainingMode.TRAINING
        )

        print(f"ONNX model successfully exported to: {output_path}")
        print("You can now visualize this file using Netron (https://netron.app/).")

    except Exception as e:
        print(f"An error occurred during ONNX export: {e}")
        import traceback
        traceback.print_exc()



