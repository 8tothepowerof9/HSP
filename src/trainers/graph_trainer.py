import time
import torch
import pandas as pd
from .base import BaseTrainer
from .utils import EarlyStopping, seconds_to_minutes_str
from .config import *


class GraphTrainer(BaseTrainer):
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
        super().__init__(
            model, loss_fn, optimizer, epochs, scheduler, device, min_lr, save
        )

    def _train_epoch(self, dataloader):
        start = time.time()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        total_loss = 0

        # Metrics
        accuracy = self.metrics["accuracy"]
        f1 = self.metrics["f1"]
        accuracy.reset()
        f1.reset()

        self.model.train()
        for batch, data in enumerate(dataloader):
            data = data.to(self.device)
            output = self.model(data)
            loss = self.loss_fn(output, data.y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            output = output.argmax(dim=1)

            accuracy(output, data.y)
            f1(output, data.y)

            # Logging
            if batch % LOG_INTERVAL == 0:
                current = batch * len(data)
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

        avg_loss = total_loss / num_batches
        accuracy_score = accuracy.compute().item()
        f1_score = f1.compute().item()

        # Save metrics
        lr = self.optimizer.param_groups[0]["lr"]
        self.log["loss"].append(avg_loss)
        self.log["accuracy"].append(accuracy_score)
        self.log["f1"].append(f1_score)
        self.log["lr"].append(lr)

        end = time.time()

        print(
            f"Train summary: [{end-start:.3f}]: \n Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy_score:.4f} | F1: {f1_score:.4f} | lr: {lr}"
        )

        return end - start

    def _eval_epoch(self, dataloader):
        start = time.time()
        num_batches = len(dataloader)
        total_loss = 0

        # Metrics
        accuracy = self.metrics["accuracy"]
        f1 = self.metrics["f1"]
        accuracy.reset()
        f1.reset()

        self.model.eval()

        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                output = self.model(data)
                loss = self.loss_fn(output, data.y)

                # Metrics
                total_loss += loss.item()
                output = output.argmax(dim=1)
                accuracy(output, data.y)
                f1(output, data.y)

        avg_loss = total_loss / num_batches
        accuracy_score = accuracy.compute().item()
        f1_score = f1.compute().item()

        # Save metrics
        self.log["val_loss"].append(avg_loss)
        self.log["val_accuracy"].append(accuracy_score)
        self.log["val_f1"].append(f1_score)

        end = time.time()

        print(
            f"Validation summary: [{end-start:.3f}]: \n Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy_score:.4f} | F1: {f1_score:.4f}"
        )

        return end - start

    def fit(self, train_dataloader, val_dataloader):
        if self.checkpoint_loaded:
            print("--- Loaded checkpoint ---")

        print("--- Dataset loaded. Start training---")

        stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}\n")
            train_time = self._train_epoch(train_dataloader)
            val_time = self._eval_epoch(val_dataloader)

            if epoch > 0:
                print(
                    "\n[Approximate time remaining]: ",
                    seconds_to_minutes_str(
                        (train_time + val_time) * (self.epochs - epoch - 1)
                    ),
                    "\n",
                )

            # Get lr
            lr = self.optimizer.param_groups[0]["lr"]
            if lr > self.min_lr:
                self.scheduler.step()
            if stopper.early_stop(self.log["val_loss"][-1], self.model, epoch + 1):
                print("Early Stopped!")
                break

        # Save model and log
        if self.save:

            checkpoint = {
                "model": stopper.best_model_state,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            }

            torch.save(checkpoint, f"{CHECKPOINT_PATH}/{self.model.name}.pt")
            print(f"--- Best model from {stopper.best_model_epoch} saved ---")

            pd.DataFrame(self.log).to_csv(
                f"{LOG_PATH}/{self.model.name}.csv", index=False
            )
            print("--- Log saved ---")

        print("--- Training finished ---")

        return self.log
