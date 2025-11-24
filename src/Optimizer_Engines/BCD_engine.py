import random
import time
from typing import List, Sequence, Tuple

import torch

import src.Buliding_Units.Model_frames as Model_frames

def log_progress(frame, round_idx, total_time, optimize_block_time, log_deviation=False):
    """Logs performance on one batch from big_train_loader with optional deviation info."""
    with torch.no_grad():
        for batch in frame.big_train_loader:
            inputs, labels = batch
            inputs = inputs.to(frame.device)
            labels = labels.to(frame.device)
            outputs = frame.center_model(inputs)
            loss = frame.criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).float().mean().item()
            # Calculate compute time share (ensuring we avoid division by zero)
            compute_share = round(optimize_block_time / total_time, 2) if total_time > 0 else 0
            message = f"Time {round_idx/1000}: Energy = {loss.item()}, Accuracy = {round(accuracy, 2)}, ComputeTimeShare = {compute_share}"
            if log_deviation:
                frame.compute_deviation()
                deviation = sum(frame.param_deviation) / len(frame.param_deviation) if frame.param_deviation else 0
                message += f", Deviation = {deviation}"
            frame.reporter.log(message)
            frame.loss_history.append(loss.item())
            # Update last known metrics for stopper checks
            frame.last_loss = loss.item()
            frame.last_accuracy = accuracy

            
            break
            

def perform_communication(frame, communicate_func):
    """Executes a communication phase using the provided function and returns the elapsed time."""
    start = time.time()
    communicate_func()
    return time.time() - start


def _draw_minibatches(dataset: Sequence[Tuple[torch.Tensor, torch.Tensor]], steps: int):
    """Sample `steps` micro-batches from the cached training loader."""
    if steps <= 0:
        return []
    data_len = len(dataset)
    if data_len == 0:
        raise ValueError("Training dataset is empty; cannot run block update.")
    if steps <= data_len:
        indices = random.sample(range(data_len), steps)
    else:
        indices = [random.randrange(data_len) for _ in range(steps)]
    return [dataset[idx] for idx in indices]


def _block_distance_sq(block_params: List[torch.nn.Parameter], reference_params: List[torch.Tensor]) -> torch.Tensor:
    """Return the squared Euclidean distance between current and reference block parameters."""
    if not block_params:
        device = reference_params[0].device if reference_params else torch.device("cpu")
        return torch.zeros(1, device=device)
    device = block_params[0].device
    distance = torch.zeros(1, device=device)
    for param, ref in zip(block_params, reference_params):
        distance += torch.sum((param - ref) ** 2)
    return distance


def _stationarity_metrics(
    model: torch.nn.Module,
    block_params: List[torch.nn.Parameter],
    reference_params: List[torch.Tensor],
    criterion,
    lambda_i: float,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[float, float]:
    """Compute gradient norm of the proximal subproblem and distance from the reference point."""
    model.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    prox_penalty = _block_distance_sq(block_params, reference_params)
    objective = loss + 0.5 * lambda_i * prox_penalty
    grads = torch.autograd.grad(objective, block_params, retain_graph=False, allow_unused=False)

    grad_norm_sq = torch.zeros(1, device=objective.device)
    for grad in grads:
        grad_norm_sq += grad.detach().pow(2).sum()
    grad_norm = torch.sqrt(grad_norm_sq + 1e-12).item()

    with torch.no_grad():
        diff_norm = torch.sqrt(_block_distance_sq(block_params, reference_params) + 1e-12).item()
    return grad_norm, diff_norm


def solve_proximal_block(
    model: torch.nn.Module,
    block_index: int,
    optimizer: torch.optim.Optimizer,
    local_steps: int,
    dataset: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    criterion,
    lambda_i: float,
    accuracy_factor: float | None,
) -> None:
    """Approximately solve the proximal block subproblem described in Algorithm 1."""
    block_params = list(model.blocks[block_index].parameters())
    if not block_params or local_steps <= 0:
        return

    reference_params = [param.detach().clone() for param in block_params]
    minibatches = _draw_minibatches(dataset, local_steps)

    model.train()
    for inputs, labels in minibatches:
        if hasattr(inputs, "device") and inputs.device != device:
            inputs = inputs.to(device, non_blocking=True)
        if hasattr(labels, "device") and labels.device != device:
            labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        task_loss = criterion(outputs, labels)
        prox_penalty = _block_distance_sq(block_params, reference_params)
        loss = task_loss + 0.5 * lambda_i * prox_penalty
        loss.backward()
        optimizer.step()

        if accuracy_factor is None or accuracy_factor < 0:
            continue

        grad_norm, diff_norm = _stationarity_metrics(
            model,
            block_params,
            reference_params,
            criterion,
            lambda_i,
            inputs,
            labels,
        )
        if diff_norm == 0:
            continue
        if grad_norm <= accuracy_factor * lambda_i * diff_norm:
            break


def train_blockwise_sequential(frame: Model_frames.Disributed_frame, share_of_active_workes: float = -1) -> None:
    """Run the proximal block coordinate descent loop described in Algorithm 1."""
    device = frame.device
    block_items = list(frame.distributed_models.values())
    if not block_items:
        raise ValueError("No distributed models were registered; cannot run blockwise training.")
    block_count = len(block_items)
    block_lambdas = frame.block_lambdas
    accuracy_factor = frame.block_accuracy_factor

    if isinstance(frame.train_loader, list):
        cached_batches = frame.train_loader
    else:
        cached_batches = list(frame.train_loader)

    frame.reporter.log("Started proximal block coordinate descent training loop")
    frame.reporter.log(
        f"Total rounds: {frame.rounds}, blocks: {block_count}, gamma={frame.gamma}, "
        f"lambda_range=[{min(block_lambdas):.3g}, {max(block_lambdas):.3g}]"
    )

    total_time_start = time.time()
    time_spent_in_optimize_block = 0.0
    iteration = 0

    for round_idx in range(frame.rounds):
        if share_of_active_workes == -1:
            active_indices = list(range(block_count))
        else:
            share = max(0.0, min(1.0, float(share_of_active_workes)))
            num_active = max(1, int(share * block_count))
            num_active = min(num_active, block_count)
            active_indices = random.sample(range(block_count), num_active)

        for block_idx in active_indices:
            model, optimizer = block_items[block_idx]
            lambda_i = block_lambdas[block_idx]
            optimize_block_start = time.time()
            solve_proximal_block(
                model=model,
                block_index=block_idx,
                optimizer=optimizer,
                local_steps=frame.K,
                dataset=cached_batches,
                device=device,
                criterion=frame.criterion,
                lambda_i=lambda_i,
                accuracy_factor=accuracy_factor,
            )
            time_spent_in_optimize_block += time.time() - optimize_block_start

        perform_communication(frame, frame.communicate_withDelay)

        if round_idx % frame.H.get("measurement_sampling_rate", 1) == 1:
            frame.run_measurmentUnits()

        total_time = time.time() - total_time_start
        if round_idx % frame.H["report_sampling_rate"] == 0:
            log_progress(frame, round_idx, total_time, time_spent_in_optimize_block, log_deviation=True)
            if hasattr(frame, "stopper_units") and frame.stopper_units:
                if any(stopper.should_stop(frame, frame.last_loss, frame.last_accuracy) for stopper in frame.stopper_units):
                    frame.reporter.log("Early stopping triggered.")
                    break

        iteration += 1

    total_time = time.time() - total_time_start
    frame.reporter.log(f"Training completed in {total_time:.2f}s over {iteration} rounds")
    print(f"Finished Training in {total_time:.2f}s over {iteration} rounds")

def train_entire(frame):
    """
    Trains the entire network using a single optimizer, with progress reporting based on iterations.

    Args:
        frame: The Classifier instance containing model, optimizer, stopper_units, etc.
    """
    center_model = frame.center_model.to(frame.device)
    criterion = frame.criterion
    optimizer = frame.optimizers[0]
    device = frame.device
    reporter = frame.reporter  # Assuming the classifier has a reporter attribute

    center_model.train()  # Set the center_model to training mode

    reporter.log("Started the training loop")
    reporter.log(f"Total epochs: {frame.rounds}")
    total_time_start = time.time()
    iteration = 0
    full_break = False

    while not full_break:
        for inputs, labels in frame.train_loader:

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = center_model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            iteration += 1

            # Measurement sampling based on new sampling rate
            if iteration % frame.H.get("measurement_sampling_rate", 1) == 1:
                frame.run_measurmentUnits()

            # Reporting sampling based on report_sampling_rate
            if iteration % frame.H["report_sampling_rate"] == 0:
                total_time = time.time() - total_time_start
                log_progress(frame, iteration, total_time, 0, log_deviation=False)
                if hasattr(frame, "stopper_units") and frame.stopper_units:
                    if any(stopper.should_stop(frame, frame.last_loss, frame.last_accuracy) for stopper in frame.stopper_units):
                        reporter.log("Early stopping triggered.")
                        full_break = True
                        break

            if iteration >= frame.rounds:
                full_break = True
                break

        

    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    reporter.log(f"Training completed in {total_time:.2f}s over {iteration} rounds")
    print(f"Finished Training in {total_time:.2f}s over {iteration} rounds")
