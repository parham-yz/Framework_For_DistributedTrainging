import torch
import src.Buliding_Units.Model_frames as Model_frames
import concurrent.futures
import time  # Import the time module
import multiprocessing
from src.Buliding_Units.StopperUnit import StopperUnit
import random

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


def optimzie_block(model, optimizer, K, data_set, device, criterion):
    """
    Trains a copy of the model for K local steps using the given optimizer.
    
    Args:
        model: The model to copy and train.
        optimizer: The optimizer tied to a specific parameter group.
        K: Number of local optimization steps.
        train_loader: DataLoader for training data.
        device: Device to run the training on (e.g., CUDA).
        criterion: Loss function.
    
    Returns:
        The updated model copy after K steps.
    """
    # Ensure model is on the correct device (only performed once per call).
    model.to(device)

    # Perform K local steps
    data_iter = iter(data_set)
    for _ in range(K):
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(data_set)
            inputs, labels = next(data_iter)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Clear all gradients (including frozen blocks) once per step
        model.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
    
    return model


# Set the start method for multiprocessing to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

def train_blockwise_distributed(frame:Model_frames.Disributed_frame):
    pass

def train_blockwise_sequential(frame: Model_frames.Disributed_frame, share_of_active_workes=-1):
    """
    Trains the model with an inner loop (round) calling train_block for each optimizer.

    Args:
        frame: The Classifier instance containing model, optimizers, stopper_units, etc.
        number_of_active_workes: Number of randomly selected active workers to use per round.
                                  If -1, all workers are used.
    """
    device = frame.device
    iteration = 0

    frame.reporter.log("Started the training loop")
    frame.reporter.log(f"Total rounds: {frame.rounds}, number of blocks: {len(frame.distributed_models)}")

    # Initialize timing measurements
    total_time_start = time.time()
    time_spent_in_optimize_block = 0
    time_spent_in_communicate = 0

    for round_idx in range(frame.rounds):
        # New: Randomly select M active workers for this round
        all_blocks = list(frame.distributed_models.keys())
        number_of_active_blocks = int(share_of_active_workes*len(all_blocks))

        if share_of_active_workes == -1 or number_of_active_blocks >= len(all_blocks):
            active_blocks = all_blocks
        else:
            active_blocks = random.sample(all_blocks, number_of_active_blocks)
        
        # Step 1: Call the optimization function only on the selected active workers
        for block_name in active_blocks:
            model, optimizer = frame.distributed_models[block_name]
            optimize_block_start = time.time()  # Start timing
            optimzie_block(
                model,
                optimizer,
                frame.K,
                frame.train_loader,
                device,
                frame.criterion
            )
            time_spent_in_optimize_block += time.time() - optimize_block_start  # Accumulate time

        # Step 2: Update the main model's blocks with the updated blocks using helper function.
        time_spent_in_communicate += perform_communication(frame, frame.communicate_withDelay)
        # Measurement sampling based on new sampling rate
        if round_idx % frame.H.get("measurement_sampling_rate", 1) == 1:
            frame.run_measurmentUnits()

        total_time = time.time() - total_time_start
        # Log progress every report_sampling_rate rounds using the helper function (with deviation)
        if round_idx % frame.H["report_sampling_rate"] == 0:
            log_progress(frame, round_idx, total_time, time_spent_in_optimize_block, log_deviation=True)
            # Check stopper condition after logging progress using stopper_units attribute
            if hasattr(frame, "stopper_units") and frame.stopper_units:
                if any(stopper.should_stop(frame, frame.last_loss, frame.last_accuracy) for stopper in frame.stopper_units):
                    frame.reporter.log("Early stopping triggered.")
                    break

        iteration += 1

    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    frame.reporter.log(f"Training completed in {total_time:.2f}s over {iteration} rounds")
    print(f"Finished Training in {total_time:.2f}s over {iteration} rounds")

def train_entire(frame):
    """
    Trains the entire network using a single optimizer, with progress reporting based on iterations.

    Args:
        frame: The Classifier instance containing model, optimizer, stopper_units, etc.
    """
    center_model = frame.center_model.to(frame.device)
    train_loader = frame.train_loader
    criterion = frame.criterion
    optimizer = frame.optimizers[0]
    device = frame.device
    reporter = frame.reporter  # Assuming the classifier has a reporter attribute

    center_model.train()  # Set the center_model to training mode

    reporter.log("Started the training loop")
    reporter.log(f"Total epochs: {frame.rounds}")
    total_time_start = time.time()
    iteration = 0
    data_iter = iter(train_loader)

    for round_idx in range(frame.rounds):
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            inputs, labels = next(data_iter)

        inputs = inputs.to(device)
        labels = labels.to(device)

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
            log_progress(frame, round_idx, total_time, 0, log_deviation=False)
            if hasattr(frame, "stopper_units") and frame.stopper_units:
                if any(stopper.should_stop(frame, frame.last_loss, frame.last_accuracy) for stopper in frame.stopper_units):
                    reporter.log("Early stopping triggered.")
                    break

    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    reporter.log(f"Training completed in {total_time:.2f}s over {iteration} rounds")
    print(f"Finished Training in {total_time:.2f}s over {iteration} rounds")
