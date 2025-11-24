"""Entry point for running distributed/blockwise training jobs."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

import src.Buliding_Units.Model_frames as Model_frames
import src.Buliding_Units.StopperUnit as StopperUnit
import src.Optimizer_Engines.BCD_engine as BCD_engine
import src.utils as utils

# Default hyper-parameters reflect a safe CIFAR10 baseline.
DEFAULT_HYPERPARAMETERS: Dict[str, Any] = {
    "step_size": 0.01,
    "batch_size": 128,
    "rounds": 1000,
    "dataset_name": "cifar10",
    "cuda_core": 0,
    "training_mode": "blockwise_sequential",
    "report_sampling_rate": 5,
    "measurement_sampling_rate": 399,
    "model": "ResNet34",
    "config": "[32, 32, 32, 32, 32]",
    "K": 1,
    "communication_delay": 0,
    "n_workers": 1,
    "beta": 0.5,
    "gamma": 1.0,
    "prox_lambda": 0.0,
    "block_accuracy_factor": 0.5,
    "reports_dir": "default_path",
}

# Handy presets that can be selected by overriding the defaults via CLI args:
#   • Regression: --dataset_name california_housing --model neural_net --config "[16]" --K 150
#   • Blockwise ResNet: --model ResNet34-bi --config "[128, 64, 32, 32, 32, 32, 16, 16]"
#   • Plot export only: --training_mode ploting --rounds 0 --reports_dir my_tag
# Keeping them here acts like a cookbook and satisfies the "comment everything" guidance.

TRAINING_DISPATCH: Dict[str, Callable[[Any], None]] = {
    "blockwise_sequential": BCD_engine.train_blockwise_sequential,
    "entire": BCD_engine.train_entire,
}


def _initialize_frame(hyperparameters: Dict[str, Any]) -> Any:
    """Construct the model frame, attach measurement units, and set stopper logic."""
    measurement_units = utils.parse_measurement_units()
    start = time.perf_counter()
    frame = Model_frames.generate_ModelFrame(hyperparameters)
    frame.set_measure_units(measurement_units)
    frame.set_stopper_units([StopperUnit.AccuracyTargetStopper(0.9)])
    init_elapsed = time.perf_counter() - start
    frame.reporter.log(f"Frame initialization took {init_elapsed:.2f} seconds")
    return frame


def _train(frame: Any, training_mode: str, hyperparameters: Dict[str, Any]) -> None:
    """Dispatch training based on the requested mode."""
    if training_mode == "ploting":
        utils.export_pytorch_to_onnx(
            frame.center_model,
            frame.input_shape,
            f"reports/{hyperparameters['reports_dir']}/model.onnx",
        )
        return

    trainer = TRAINING_DISPATCH.get(training_mode)
    if trainer is None:
        raise ValueError(f"Invalid training mode: {training_mode}")
    trainer(frame)


def main() -> None:
    """Parse CLI arguments, bootstrap the model frame, and kick off training."""
    hyperparameters = utils.parse_arguments(DEFAULT_HYPERPARAMETERS.copy())
    frame = _initialize_frame(hyperparameters)

    start_time = time.perf_counter()
    total_params = sum(p.numel() for p in frame.center_model.parameters())
    frame.reporter.log(f"Total number of parameters: {total_params:,}")

    _train(frame, hyperparameters["training_mode"], hyperparameters)

    elapsed = time.perf_counter() - start_time
    frame.reporter.log(f"Training completed in {elapsed:.2f} seconds")
    frame.reporter.terminate()


if __name__ == "__main__":
    main()
