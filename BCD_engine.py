import torch
import Model_frames
import concurrent.futures
import time  # Import the time module
import multiprocessing



def optimzie_block(model, optimizer, K, train_loader, device, criterion):
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
    
    
    # Perform K local steps
    for _ in range(K):
        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            break  # Process only one batch per step for simplicity
    
    return model






# Set the start method for multiprocessing to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

def train_blockwise(frame):
    """
    Trains the model with an inner loop (round) calling optimzie_block for each optimizer in parallel using multiprocessing.
    
    Args:
        frame: The Classifier_frame_blockwise instance containing model, optimizers, etc.
    """
    device = frame.device
    total_rounds = frame.rounds

    frame.reporter.log("Started the training loop")

    # Initialize timing measurements
    total_time_start = time.time()
    time_spent_in_optimize_block = 0
    time_spent_in_communicate = 0

    # Create a persistent pool of workers
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for round_idx in range(total_rounds):
            futures = []
            optimize_block_start = time.time()  # Start timing before launching processes

            for block_name, (model, optimizer) in frame.distributed_models.items():
                # Serialize model and optimizer
                model.share_memory()  # Prepare model for being sent to another process
                future = executor.submit(
                    optimzie_block,
                    model,
                    optimizer,
                    frame.K,
                    frame.train_loader,
                    device,
                    frame.criterion
                )
                futures.append(future)

            # Wait for all processes to complete
            concurrent.futures.wait(futures)

            # Accumulate time spent in optimize_block
            time_spent_in_optimize_block += time.time() - optimize_block_start

            # Communication phase
            communicate_start = time.time()
            frame.communicate()
            time_spent_in_communicate += time.time() - communicate_start

        
        # Step 3: Log progress every 100 rounds  
        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        with torch.no_grad():  # Disable gradient computation
            if round_idx % frame.H["report_sampling_rate"] == 0:
                for batch in frame.big_train_loader:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = frame.center_model(inputs)
                    loss = frame.criterion(outputs, labels)
                    _, predicted = torch.max(outputs, 1)
                    accuracy = (predicted == labels).float().mean().item()
                    frame.reporter.log(f"Time {(round_idx)/(1000)}: Energy = {loss.item()}, Accuracy = {round(accuracy, 2)}, ComputeTimeShare = {round(time_spent_in_optimize_block/total_time, 2)}")  # Now using Reporter
                    frame.loss_history.append(loss.item())
                    break  # Evaluate on one batch

    frame.reporter.log("Training completed")



def train_blockwise_sequential(frame: Model_frames.ImageClassifier_frame_blockwise):
    """
    Trains the model with an inner loop (round) calling train_block for each optimizer.
    
    Args:
        classifier: The Classifier instance containing model, optimizers, etc.
    """
    device = frame.device
    iteration = 0

    frame.reporter.log("Started the training loop")  # Now using Reporter

    # Initialize timing measurements
    total_time_start = time.time()
    time_spent_in_optimize_block = 0
    time_spent_in_communicate = 0

    for round_idx in range(frame.rounds):
        # Step 1: Call train_block for each block's optimizer
        for block_name in frame.distributed_models.keys():
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

        # Step 2: Update the main model's blocks with the updated blocks
        communicate_start = time.time()  # Start timing
        frame.communicate()
        time_spent_in_communicate += time.time() - communicate_start  # Accumulate time

        total_time_end = time.time()
        total_time = total_time_end - total_time_start

        # Step 3: Log progress every 100 rounds
        with torch.no_grad():  # Disable gradient computation
            if round_idx % frame.H["report_sampling_rate"] == 0:
                for batch in frame.big_train_loader:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = frame.center_model(inputs)
                    loss = frame.criterion(outputs, labels)
                    _, predicted = torch.max(outputs, 1)
                    accuracy = (predicted == labels).float().mean().item()
                    frame.compute_deviation()
                    param_deviation = sum(frame.param_deviation)/len(frame.param_deviation)
                    frame.reporter.log(f"Time {(round_idx)/(1000)}: Energy = {loss.item()}, Accuracy = {round(accuracy, 2)}, ComputeTimeShare = {round(time_spent_in_optimize_block/total_time, 2)}, Deviation = {param_deviation}")  # Now using Reporter
                    frame.loss_history.append(loss.item())
                    break  # Evaluate on one batch

        iteration += 1




    frame.reporter.log("Training completed")  # Now using Reporter

def train_entire(frame):
    """
    Trains the entire network using a single optimizer, with progress reporting based on iterations.

    Args:
        classifier: The Classifier instance containing model, optimizer, etc.
    """
    model = frame.model.to(frame.device)
    train_loader = frame.train_loader
    criterion = frame.criterion
    optimizer = frame.optimizers[0]
    device = frame.device
    reporter = frame.reporter  # Assuming the classifier has a reporter attribute

    model.train()  # Set the model to training mode

    reporter.log("Started the training loop")
    iteration = 0

    for round_idx in range(frame.rounds):
        inputs, labels = next(iter(train_loader))
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        iteration += 1
        with torch.no_grad():
            if iteration % frame.H["report_sampling_rate"] == 0:
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == labels).float().mean().item()
                reporter.log(f"Time {(round_idx)/(1000)}: Energy = {loss.item()}, Accuracy = {round(accuracy, 2)}")
                frame.loss_history.append(loss.item())

    reporter.log("Training completed")
    print('Finished Training')