from abc import ABC, abstractmethod
import os
import torch
import copy


class MeasurementUnit(ABC):
    """
    Abstract class for modules used in a training frame that measure some property of the models.
    
    The `measure` method should be implemented by subclasses to compute a numerical metric
    given a list of models. Additionally, this class initializes a reporter that logs measurement
    values to a file located in the 'measurements' folder.
    """
    
    def __init__(self, measure_name):
        # Ensure that the "measurements" folder exists
        measurements_folder = "measurements"
        os.makedirs(measurements_folder, exist_ok=True)
        
        # Set the file path for the log file using measure_name
        file_name = f"{measure_name.replace(' ', '_').lower()}_log.txt"
        self.file_path = os.path.join(measurements_folder, file_name)
        
        # Initialize the reporter by opening the file in append mode
        self.reporter = open(self.file_path, "a")
        
        # Write initial information to the log file
        pid = os.getpid()
        self.reporter.write(f"Process ID: {pid}\n")
        self.reporter.write(f"Measurement Description: {measure_name}\n")
        self.reporter.write("\n")
        self.reporter.flush()

    @abstractmethod
    def measure(self, frame) -> float:
        """
        Compute and return a measurement given a list of model instances.
        
        Args:
            models (list): A list of model objects.
        
        Returns:
            float: The measurement value.
        """
        pass
    
    def log_measurement(self, measurement: float):
        """
        Log the measured value to the reporter file.
        
        Args:
            measurement (float): The measurement to be logged.
        """
        self.reporter.write(f"{measurement}\n")
        self.reporter.flush()
    
    def close(self):
        """
        Close the reporter file handle if it is open.
        """
        if not self.reporter.closed:
            self.reporter.close()
    
    def __del__(self):
        self.close()



class Working_memory_usage(MeasurementUnit):
    """
    Measurement unit that computes the memory usage of the model and optimizer.
    It calculates the total memory used by the model parameters and optimizer state,
    and returns the cumulative memory usage in megabytes (MB).
    """
    
    def __init__(self):
        super().__init__("Memory Usage Measurement")
    
    def measure(self, frame) -> float:
        """
        Compute the memory usage for each client.
        
        For distributed frames, it computes the memory usage for each client (i.e., each
        entry in 'distributed_models') by summing the memory used by its model and optimizer.
        For non-distributed frames, it computes the memory usage for the central model (frame.model)
        and for each optimizer in frame.optimizers.
        
        Returns:
            float: Total memory usage in megabytes (MB).
        """
        total_memory_bytes = 0
        
        # Check if the frame uses distributed models (i.e., multiple clients)
        if hasattr(frame, "distributed_models") and frame.distributed_models:
            for client, (model, optimizer) in frame.distributed_models.items():
                total_memory_bytes += self._get_model_memory(model)
                total_memory_bytes += self._get_optimizer_memory(optimizer)

            total_memory_bytes = total_memory_bytes/len(frame.distributed_models)

        else:
            # Non-distributed case: use frame.model (if available) and
            # iterate through the list of optimizers.
            if hasattr(frame, "model"):
                total_memory_bytes += self._get_model_memory(frame.model)
            if hasattr(frame, "optimizers"):
                for optimizer in frame.optimizers:
                    total_memory_bytes += self._get_optimizer_memory(optimizer)
        
        # Convert the total memory from bytes to megabytes.
        total_memory_mb = total_memory_bytes / (1024 * 1024)
        return total_memory_mb
    
    def _get_model_memory(self, model) -> int:
        """
        Calculate the memory used by the model's parameters.
        
        Returns:
            int: Memory in bytes.
        """
        memory = 0
        for param in model.parameters():
            memory += param.nelement() * param.element_size()
        return memory
    
    def _get_optimizer_memory(self, optimizer) -> int:
        """
        Calculate the memory used by the optimizer's state.
        
        Returns:
            int: Memory in bytes.
        """
        memory = 0
        for state in optimizer.state.values():
            memory += self._get_memory_of_obj(state)
        return memory



class Hessian_measurement(MeasurementUnit):
    """
    Measurement unit that approximates the full Hessian of the loss function for the model using 
    Hessian-vector products (HVPs) with finite differences. A single batch from the training loader 
    is used to perform the approximation. The full Hessian matrix is written to the log with the appropriate 
    begin/end markers.
    """
    def __init__(self):
        super().__init__("Hessian Measurement")
    
    
    def measure(self, frame) -> float:
        model = frame.center_model
        device = next(model.parameters()).device
        data_iter = iter(frame.train_loader)
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            return 0.0
        inputs, targets = inputs.to(device), targets.to(device)
        
        criterion = getattr(frame, "criterion", torch.nn.MSELoss())

        hessian = _compute_hessian(model, inputs, targets, criterion)
        ls = _get_block_parameter_counts(model)
        hessian_blocks = _decompose_matrix_to_blocks(hessian,ls)
        
        diagonal_elements = torch.diag(hessian)
        diag_row_norm = [sum(hessian_blocks[i]) for i in range(len(hessian_blocks))]
        off_diagonal_elements = hessian - torch.diag(diagonal_elements)
        frob_norm_off_diagonal = torch.norm(off_diagonal_elements).item()
        return frob_norm_off_diagonal / frob_norm_diagonal


def _compute_hessian(model, inputs, targets, criterion, epsilon=1e-4):
    """
    Computes the Hessian matrix of the loss with respect to the model parameters using finite differences.
    
    Inputs:
      model:        The neural network model (a torch.nn.Module).
      inputs:       Tensor of input data to the model.
      targets:      Tensor of target outputs corresponding to the inputs.
      criterion:    Loss function used to compute the loss between model outputs and targets.
      epsilon:      Small perturbation value for finite difference approximation (default is 1e-4).
    """
    device = next(model.parameters()).device
    
    # Save original parameters as a flattened vector
    original_vector = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    n_params = original_vector.numel()
    
    # Compute baseline loss and its gradient
    model.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    grad_vec = torch.nn.utils.parameters_to_vector(grad).detach()
    
    # Initialize Hessian matrix
    H = torch.zeros(n_params, n_params, device=device)
    for i in range(n_params):
        perturb = torch.zeros_like(original_vector)
        perturb[i] = epsilon
        perturbed_vector = original_vector + perturb
        # Update model parameters with the perturbed vector
        torch.nn.utils.vector_to_parameters(perturbed_vector, model.parameters())
        
        model.zero_grad()
        outputs_perturbed = model(inputs)
        loss_perturbed = criterion(outputs_perturbed, targets)
        grad_perturbed = torch.autograd.grad(loss_perturbed, model.parameters(), create_graph=False)
        grad_perturbed_vec = torch.nn.utils.parameters_to_vector(grad_perturbed).detach()
        
        hvp = (grad_perturbed_vec - grad_vec) / epsilon
        H[:, i] = hvp
    
    # Restore original parameters
    torch.nn.utils.vector_to_parameters(original_vector, model.parameters())
    return H

def _decompose_matrix_to_blocks(H, l):
    """
    Decomposes a square PyTorch tensor H into block tensors based on a list of integers l.

    Args:
        H (torch.Tensor): The n*n square PyTorch tensor to decompose.
                          Requires H.dim() == 2.
        l (list of int): A list of positive integers [l1, l2, ...]
                         representing the dimensions of the square diagonal blocks.
                         The sum of elements in l must equal n.

    Returns:
        list of lists of torch.Tensor:
            A nested list representing the block tensor H'.
            H'[i][j] is the block (a torch.Tensor) at the i-th block row
            and j-th block column.
            Returns None if the input is invalid.

    Raises:
        ValueError: If inputs are invalid (e.g., H is not a 2D tensor, H is not square,
                    sum of l != n, l contains non-positive integers).
        TypeError: If H is not a PyTorch tensor.
    """
    # --- Input Validation ---
    if not isinstance(H, torch.Tensor):
      raise TypeError(f"Input H must be a PyTorch tensor (got {type(H)}).")

    if H.dim() != 2:
        raise ValueError(f"Input tensor H must be 2-dimensional (got {H.dim()} dimensions).")

    n_rows, n_cols = H.shape
    if n_rows != n_cols:
        raise ValueError(f"Input tensor H must be square (got shape {H.shape}).")

    n = n_rows

    if not isinstance(l, list) or not all(isinstance(x, int) and x > 0 for x in l):
        raise ValueError("Input l must be a list of positive integers.")

    if sum(l) != n:
        raise ValueError(f"The sum of elements in l ({sum(l)}) must equal the dimension of H ({n}).")

    # --- Block Decomposition ---
    block_matrix = []
    # Use torch.cumsum for indices calculation
    indices = torch.tensor([0] + l).cumsum(dim=0) # Start indices [0, l1, l1+l2, ...]

    for i in range(len(l)):
        block_row = []
        rows = slice(indices[i].item(), indices[i+1].item()) # Row slice for the i-th block row

        for j in range(len(l)):
            cols = slice(indices[j].item(), indices[j+1].item()) # Column slice for the j-th block col

            # Extract the block using tensor slicing
            block = H[rows, cols]
            block_row.append(block)

        block_matrix.append(block_row)

    return block_matrix

def _get_block_parameter_counts(model):
  """
  Calculates the number of parameters for each layer in a PyTorch model.

  Args:
    model: A PyTorch nn.Module object.

  Returns:
    A list of integers, where each integer represents the number of
    parameters in the corresponding layer of the model.
  """
  param_counts = []
  for block in model.blocks:
    num_params = sum(p.numel() for p in block.parameters())
    param_counts.append(num_params)
  return param_counts