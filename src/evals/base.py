from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    def __init__(self, model, log, device, save=True):
        self.model = model
        self.log = log
        self.device = device
        self.save = save

    @abstractmethod
    def evaluate(self, dataloader):
        """Evaluate on a dataset and return the metrics"""
        raise NotImplementedError

    @abstractmethod
    def realtime_predict(self):
        """Open the webcam and predict in real-time"""
        raise NotImplementedError

    @abstractmethod
    def static_predict(self, image_path):
        """Predict on a static image"""
        raise NotImplementedError

    @abstractmethod
    def plot_history(self):
        """Plot the training history"""
        raise NotImplementedError

    @abstractmethod
    def confusion_matrix(self, dataloader):
        """Plot the confusion matrix"""
        raise NotImplementedError
