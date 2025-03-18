from abc import ABC, abstractmethod
from torchmetrics import Accuracy, F1Score, AUROC


class BaseEvaluator(ABC):
    def __init__(self, model, log, device, save=True):
        self.model = model
        self.log = log
        self.device = device
        self.save = save
        task = "binary"
        self.metrics = {
            "accuracy": Accuracy(task=task).to(device),
            "f1": F1Score(task=task).to(device),
            "auroc": AUROC(task=task).to(device),
        }

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
