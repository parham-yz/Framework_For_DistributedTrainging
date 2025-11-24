"""Utility helpers for distributed training and experiment orchestration."""

from __future__ import annotations

import argparse
import ast
import copy
import datetime
import hashlib
import inspect
import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, Iterable, List

import torch
import torch.onnx
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.onnx import TrainingMode

import src.Buliding_Units.MeasurementUnit as MeasurementUnit


def _install_package(package_name: str) -> None:
    """Install a missing dependency in-place so CLI runs keep working."""
    logging.info("Installing missing dependency: %s", package_name)
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


# Lazily import GPU/ONNX helpers, installing them when necessary.
try:  # pragma: no cover - external dependency bootstrap
    import GPUtil  # type: ignore
except ImportError:  # pragma: no cover - runtime bootstrap path
    _install_package("GPUtil")
    import GPUtil  # type: ignore

try:  # pragma: no cover - external dependency bootstrap
    import onnx  # noqa: F401  # type: ignore
except ImportError:  # pragma: no cover - runtime bootstrap path
    _install_package("onnx")
    import onnx  # noqa: F401  # type: ignore


def _build_dataset(dataset_name: str) -> datasets.VisionDataset:
    """Return a torchvision dataset configured with consistent transforms."""
    resize_transform = transforms.Resize((32, 32))
    to_tensor_transform = transforms.ToTensor()

    name = dataset_name.lower()
    if name == "mnist":
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                resize_transform,
                to_tensor_transform,
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        return datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    if name == "cifar10":
        transform = transforms.Compose(
            [
                resize_transform,
                to_tensor_transform,
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        return datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    if name == "svhn":
        transform = transforms.Compose(
            [
                resize_transform,
                to_tensor_transform,
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        return datasets.SVHN(root="./data", split="train", download=True, transform=transform)

    raise ValueError(f"Unknown dataset name: {dataset_name}")


def get_max_batch_size(dataset_name: str, cuda_core: int = 0) -> int:
    """Estimate the largest batch that fits in GPU memory for a given dataset."""
    gpus = GPUtil.getGPUs()
    if not gpus:
        raise RuntimeError("No GPU found – required to compute batch size from device memory.")
    if cuda_core < 0 or cuda_core >= len(gpus):
        raise ValueError(f"cuda_core index {cuda_core} is out of range for detected GPUs.")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda is not available, cannot size GPU batches.")

    gpu = gpus[cuda_core]
    total_memory = int(gpu.memoryTotal * 1024 * 1024)  # Convert MiB to bytes.
    reserved_memory = int(0.1 * total_memory)  # Keep ≈10% headroom for other processes.
    available_memory = total_memory - reserved_memory

    dataset = _build_dataset(dataset_name)
    device = torch.device(f"cuda:{cuda_core}")

    # Use a single sample as a proxy for the entire batch footprint.
    sample = dataset[0][0].unsqueeze(0).to(device)
    sample_memory = sample.element_size() * sample.nelement()

    max_batch_size = available_memory // sample_memory
    max_batch_size = max(1, min(max_batch_size, len(dataset)))
    return int(max_batch_size)


def clear_gpu_memory() -> None:
    """Aggressively reset the GPU via nvidia-smi; useful when jobs OOM."""
    logging.info("Clearing GPU memory via nvidia-smi --gpu-reset")
    try:
        subprocess.run(["nvidia-smi", "--gpu-reset"], check=True)
        logging.info("Successfully cleared GPU memory.")
    except subprocess.CalledProcessError as exc:
        logging.error("Failed to clear GPU memory: %s", exc)
    except Exception as exc:  # pylint: disable=broad-except
        logging.error("Unexpected error while clearing GPU memory: %s", exc)


class Reporter:
    """Lightweight experiment reporter that logs metadata and process status."""

    def __init__(self, hyperparameters: Dict[str, Any]) -> None:
        self.hyperparameters = hyperparameters
        hyperparameters_str = json.dumps(hyperparameters, sort_keys=True)
        hash_hex = hashlib.md5(hyperparameters_str.encode(), usedforsecurity=False).hexdigest()[:8]

        reports_subdir = hyperparameters.get("reports_dir", "default_path")
        reports_path = os.path.join("reports", reports_subdir)
        self.log_filename = os.path.join(reports_path, f"R{hash_hex}.txt")
        self.reports_path = reports_path
        self.ensure_log_file()

    def ensure_log_file(self) -> None:
        """Create or reset the log file so a fresh run never appends stale info."""
        os.makedirs(self.reports_path, exist_ok=True)
        if os.path.exists(self.log_filename):
            os.remove(self.log_filename)

        with open(self.log_filename, "a", encoding="utf-8") as fh:
            pid = os.getpid()
            fh.write(f"Process ID: {pid}\n")
            fh.write("Status: alive\n")
            fh.write(f"Hyperparameters: {json.dumps(self.hyperparameters)}\n\n")

    def log(self, message: str) -> None:
        """Append a timestamped message to the report file."""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_filename, "a", encoding="utf-8") as fh:
            fh.write(f"[{current_time}] {message}\n")

    def terminate(self) -> None:
        """Mark the report file as completed so orchestration scripts can poll it."""
        self.log("Process ended")
        with open(self.log_filename, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        if len(lines) > 1 and "Status: alive" in lines[1]:
            lines[1] = "Status: dead\n"

        with open(self.log_filename, "w", encoding="utf-8") as fh:
            fh.writelines(lines)


def copy_block(source_model: torch.nn.Module, target_model: torch.nn.Module, block_index: int) -> None:
    """Copy the parameters of a specific block between two compatible models."""
    if block_index < 0 or block_index >= len(source_model.blocks):
        raise IndexError("Block index is out of range.")

    source_block = source_model.blocks[block_index]
    target_block = target_model.blocks[block_index]

    with torch.no_grad():
        for source_param, target_param in zip(source_block.parameters(), target_block.parameters()):
            target_param.data = copy.deepcopy(source_param.data).detach()


def parse_arguments(hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """Dynamically parse CLI arguments based on default hyperparameter dict."""
    parser = argparse.ArgumentParser(
        description="Dynamically process input arguments for training configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    for key, value in hyperparameters.items():
        arg_type = type(value)
        if arg_type is bool:
            parser.add_argument(
                f"--{key}",
                type=lambda x: str(x).lower() == "true",
                default=value,
                help=f"Boolean flag for {key}",
            )
        elif arg_type is list:
            parser.add_argument(
                f"--{key}",
                type=str,
                default=json.dumps(value),
                help=f"{key} (provide as a string like '[item1, item2]')",
            )
        else:
            parser.add_argument(
                f"--{key}",
                type=arg_type,
                default=value,
                help=f"{key} (type: {arg_type.__name__})",
            )

    args = parser.parse_args()
    parsed = vars(args)

    for key, value in parsed.items():
        if key not in hyperparameters:
            continue
        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
            try:
                parsed_value = ast.literal_eval(value)
                hyperparameters[key] = parsed_value if isinstance(parsed_value, list) else value
            except (ValueError, SyntaxError, TypeError):
                hyperparameters[key] = value
        elif isinstance(hyperparameters[key], list) and isinstance(value, str):
            hyperparameters[key] = hyperparameters[key]
        else:
            hyperparameters[key] = value

    return hyperparameters


def copy_model(source_model: torch.nn.Module, target_model: torch.nn.Module, device: torch.device) -> None:
    """Copy parameters from source_model to target_model on the requested device."""
    target_model.to(device)
    target_model.load_state_dict(source_model.state_dict())


def parse_measurement_units() -> List[MeasurementUnit.MeasurementUnit]:
    """Parse measurement unit names from measurements/Units.txt and instantiate them."""
    file_path = os.path.join("measurements", "Units.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Units file not found: {file_path}")

    unit_map = {
        cls.__name__: cls
        for _, cls in inspect.getmembers(MeasurementUnit, inspect.isclass)
        if issubclass(cls, MeasurementUnit.MeasurementUnit) and cls is not MeasurementUnit.MeasurementUnit
    }

    measurement_units: List[MeasurementUnit.MeasurementUnit] = []
    with open(file_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("-"):
            unit_name = stripped[1:].strip()
            if unit_name not in unit_map:
                raise ValueError(f"Unknown measurement unit: {unit_name}")
            measurement_units.append(unit_map[unit_name]())

    return measurement_units


def export_pytorch_to_onnx(
    model: torch.nn.Module,
    input_shape: Iterable[int],
    output_path: str,
    input_names: List[str] | None = None,
    output_names: List[str] | None = None,
    dynamic_axes: Dict[str, Dict[int, str]] | None = None,
    verbose: bool = False,
) -> None:
    """Export a PyTorch model to ONNX, forcing execution on CPU for reproducibility."""
    if not input_shape or not isinstance(tuple(input_shape), tuple):
        raise ValueError("`input_shape` must be provided as a tuple or list (e.g., (1, 3, 224, 224)).")

    try:
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        dummy_input = torch.randn(tuple(input_shape), requires_grad=False, device=device)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names or ["input"],
            output_names=output_names or ["output"],
            dynamic_axes=dynamic_axes,
            verbose=verbose,
            training=TrainingMode.TRAINING,
        )
        print(f"ONNX model successfully exported to: {output_path}")
        print("You can now visualize this file using Netron (https://netron.app/).")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"An error occurred during ONNX export: {exc}")
        import traceback

        traceback.print_exc()
