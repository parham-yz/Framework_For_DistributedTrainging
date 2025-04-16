from abc import ABC, abstractmethod
import os

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



class HessianMeasurement(MeasurementUnit):
    """
    Measurement unit that approximates the full Hessian of the loss function for the model using 
    Hessian-vector products (HVPs) with finite differences. A single batch from the training loader 
    is used to perform the approximation. The full Hessian matrix is written to the log with the appropriate 
    begin/end markers.
    """
    def __init__(self):
        super().__init__("Hessian Measurement")
    
    def measure(self, frame) -> float:
        import torch  # Required for tensor operations
        
        # Use the central model from the frame
        model = frame.center_model
        device = next(model.parameters()).device
        
        # Get a single batch from the training loader
        data_iter = iter(frame.train_loader)
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            return 0.0
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Get the loss function from the frame or fallback to MSELoss
        criterion = getattr(frame, "criterion", torch.nn.MSELoss())
        
        # Save the original parameters as a flattened vector
        original_vector = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
        n_params = original_vector.numel()
        epsilon = 1e-4  # finite difference step size
        
        # Compute the baseline loss and its gradient
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        grad0 = torch.autograd.grad(loss, model.parameters(), create_graph=False)
        grad0_vec = torch.nn.utils.parameters_to_vector(grad0).detach()
        
        # Initialize a Hessian matrix (n_params x n_params)
        H = torch.zeros(n_params, n_params, device=device)
        
        # Compute each column of the Hessian using finite differences
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
            
            # Approximate the i-th Hessian column via finite differences
            hvp = (grad_perturbed_vec - grad0_vec) / epsilon
            H[:, i] = hvp
        
        # Restore original parameters
        torch.nn.utils.vector_to_parameters(original_vector, model.parameters())
        
        # Write the Hessian to the log with markers before and after
        self.reporter.write("\n hessian begins:\n")
        self.reporter.write(str(H.tolist()))
        self.reporter.write("\n hessian ends \n")
        
        # Return a summary value (the Frobenius norm of the Hessian)
        frob_norm = torch.norm(H).item()
        return frob_norm
