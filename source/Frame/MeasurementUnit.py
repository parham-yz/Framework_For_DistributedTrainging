from abc import ABC, abstractmethod
import os
from Model_frames import Frame

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
    def measure(self, models: Frame) -> float:
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
