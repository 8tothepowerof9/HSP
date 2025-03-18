import os
import sys
import torch
from utils import read_config


sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from dataset import DATASET_LIST, DATALOADER_TYPE
from models import MODEL_LIST
from trainers import TRAINER_LIST

if __name__ == "__main__":
    config = read_config()

    dataset_type = DATASET_LIST[config["data"]["type"]]
    train_dataset = dataset_type(split="train", ds_dir="data")
    test_dataset = dataset_type(split="test", ds_dir="data")

    dataloader_type = DATALOADER_TYPE[config["data"]["type"]]

    train_loader = dataloader_type(
        train_dataset,
        batch_size=config["data"]["bs"],
        shuffle=True,
    )
    test_loader = dataloader_type(
        test_dataset,
        batch_size=config["data"]["bs"],
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model_type = MODEL_LIST[config["model"]["type"]]
    model = model_type(config).to(device)

    optimizer = model.get_criterion()
    criterion = model.get_optimizer()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["train"]["step_size"],
        gamma=config["train"]["gamma"],
    )

    trainer_type = TRAINER_LIST[config["data"]["type"]]
    trainer = trainer_type(
        model,
        criterion,
        optimizer,
        config["train"]["epochs"],
        scheduler,
        device,
        save=config["train"]["save"],
    )
    history = trainer.fit(train_loader, test_loader)
