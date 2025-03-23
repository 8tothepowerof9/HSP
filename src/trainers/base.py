import os
from abc import abstractmethod, ABC
import torch
import pandas as pd
from torchmetrics import Accuracy, F1Score
from dataset import NUM_LABELS
from .config import *


class BaseTrainer(ABC):
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        epochs,
        scheduler,
        device,
        min_lr=1e-6,
        save=True,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        task = "multiclass"
        self.metrics = {
            "accuracy": Accuracy(task=task, num_classes=NUM_LABELS).to(device),
            "f1": F1Score(task=task, num_classes=NUM_LABELS).to(device),
        }
        self.device = device
        self.min_lr = min_lr
        self.save = save
        self.log = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "f1": [],
            "val_accuracy": [],
            "val_f1": [],
            "lr": [],
        }
        self.epochs = epochs
        self.checkpoint_loaded = self.load_checkpoint()

    @abstractmethod
    def _train_epoch(self, dataloader):
        """Train the model on dataloader for one epoch.

        Args:
            dataloader (_type_): Train dataloader

        Raises:
            NotImplementedError: If the method is not implemented by the subclass
        """
        raise NotImplementedError

    @abstractmethod
    def _eval_epoch(self, dataloader):
        """Evaluate the model on dataloader for one epoch

        Args:
            dataloader (_type_): Validation dataloader

        Raises:
            NotImplementedError: If the method is not implemented by the subclass
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, train_dataloader, val_dataloader):
        """Train and evaluate the model for multiple epochs

        Args:
            train_dataloader (_type_): Train dataloader
            val_dataloader (_type_): Validation dataloader

        Raises:
            NotImplementedError: If the method is not implemented by the subclass
        """
        raise NotImplementedError

    def load_checkpoint(self):
        # Find if checkpoint exists, if so load and return True, if not return False
        checkpoint_path = os.path.join(CHECKPOINT_PATH, self.model.name + ".pt")
        log_path = os.path.join(LOG_PATH, self.model.name + ".csv")

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.log = pd.read_csv(log_path).to_dict()
            return True
        return False
