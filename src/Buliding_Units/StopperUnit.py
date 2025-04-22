from abc import ABC, abstractmethod
from src.Buliding_Units.Model_frames import Frame

class StopperUnit(ABC):
    @abstractmethod
    def should_stop(self, frame) -> bool:
        """
        Determines whether the training process should be stopped.

        Args:
            frame (Frame): The frame object containing the current state of the training process.

        Returns:
            bool: True if the training should be stopped, False otherwise.
        """
        pass

class EarlyStoppingStopper(StopperUnit):
    """
    A stopper unit that implements early stopping based on a specified measure.

    This class monitors the validation loss during training and stops the training
    process if the loss does not improve for a specified number of consecutive epochs
    (patience). It is useful for preventing overfitting by halting training once the
    model performance on a validation set starts to degrade.

    Usage:
        - Initialize the stopper with desired patience and minimum delta.
        - Call the `should_stop` method after each epoch with the current frame to
          determine if training should be stopped.

    Args:
        patience (int): Number of epochs to wait for an improvement in loss before stopping.
        min_delta (float): Minimum change in the monitored loss to qualify as an improvement.

    Example:
        stopper = EarlyStoppingStopper(patience=5, min_delta=0.01)
        for epoch in range(max_epochs):
            # Training logic here
            if stopper.should_stop(frame):
                print("Early stopping triggered.")
                break
    """
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def should_stop(self, frame,trainig_loss, trainig_accuracy) -> bool:
        current_loss = self.stop_measure(frame,trainig_loss, trainig_accuracy)
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
    
    @abstractmethod
    def stop_measure(self, frame:Frame, trainig_loss, trainig_accuracy) -> float:
        pass

class LossEarlyStopper(EarlyStoppingStopper):
    """
    A stopper unit that implements early stopping based on the training loss.

    This class monitors the training loss during training and stops the training
    process if the loss does not improve for a specified number of consecutive epochs
    (patience). It is useful for preventing overfitting by halting training once the
    model's performance stagnates.

    Usage:
        - Initialize the stopper with desired patience and minimum delta.
        - Call the `should_stop` method after each epoch with the current frame to
          determine if training should be stopped.

    Args:
        patience (int): Number of epochs to wait for an improvement in training loss before stopping.
        min_delta (float): Minimum change in the monitored training loss to qualify as an improvement.

    Example:
        stopper = LossEarlyStopper(patience=5, min_delta=0.01)
        for epoch in range(max_epochs):
            # Training logic here
            if stopper.should_stop(frame, training_loss, training_accuracy):
                print("Early stopping triggered.")
                break
    """
    def stop_measure(self, frame:Frame, training_loss, training_accuracy) -> float:
        return training_loss
    

class AccuracyEarlyStopper(EarlyStoppingStopper):
    """
    A stopper unit that implements early stopping based on the training accuracy.
    
    This class monitors the training accuracy during training and stops the training
    process if the accuracy does not improve (i.e., does not increase) for a specified
    number of consecutive epochs (patience). It helps in preventing overfitting by halting
    training when the accuracy plateaus.
    
    Args:
        patience (int): Number of epochs to wait for an improvement in accuracy before stopping.
        min_delta (float): Minimum change in the monitored accuracy to qualify as an improvement.
    
    Example:
        stopper = AccuracyEarlyStopper(patience=5, min_delta=0.01)
        for epoch in range(max_epochs):
            if stopper.should_stop(frame, training_loss, training_accuracy):
                print("Accuracy early stopping triggered.")
                break
    """
    def stop_measure(self, frame:Frame, training_loss, training_accuracy) -> float:
        # Invert the training accuracy so that an improvement (increase) results in a lower measure.
        return -training_accuracy
    

class TargetStopper(StopperUnit, ABC):
    """
    Base class for target-based stopping. Stops training when a specified metric reaches a threshold.
    """
    def __init__(self, threshold: float):
        self.threshold = threshold

    def should_stop(self, frame, training_loss, training_accuracy) -> bool:
        metric_value = self.get_metric(frame, training_loss, training_accuracy)
        return self.check_target(metric_value)

    @abstractmethod
    def get_metric(self, frame, training_loss, training_accuracy) -> float:
        # Extract the metric value from input parameters.
        pass

    @abstractmethod
    def check_target(self, metric_value: float) -> bool:
        # Return True if the metric_value meets the stopping condition.
        pass

class LossTargetStopper(TargetStopper):
    """
    Stops training when the training loss reaches or goes below the specified threshold.

    Args:
        threshold (float): The loss threshold; training stops when training_loss <= threshold.
    """
    def get_metric(self, frame, training_loss, training_accuracy) -> float:
        return training_loss

    def check_target(self, metric_value: float) -> bool:
        return metric_value <= self.threshold

class AccuracyTargetStopper(TargetStopper):
    """
    Stops training when the training accuracy reaches or exceeds the specified threshold.

    Args:
        threshold (float): The accuracy threshold; training stops when training_accuracy >= threshold.
    """
    def get_metric(self, frame, training_loss, training_accuracy) -> float:
        return training_accuracy

    def check_target(self, metric_value: float) -> bool:
        return metric_value >= self.threshold




