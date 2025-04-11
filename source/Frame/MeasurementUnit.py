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
    def measure(self, models: list) -> float:
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

