import os
import sys
import torch
from utils import read_config

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from dataset import DATASET_LIST, DATALOADER_TYPE
from models import MODEL_LIST
from trainers import CHECKPOINT_PATH, LOG_PATH
from evals import EVAL_LIST

if __name__ == "__main__":
    config = read_config()

    dataset_type = DATASET_LIST[config["data"]["type"]]
    train_dataset = dataset_type(split="train", ds_dir="data")
    test_dataset = dataset_type(split="test", ds_dir="data")

    dataloader_type = DATALOADER_TYPE[config["data"]["type"]]

    if config["eval"]["data"] == "train":
        loader = dataloader_type(
            train_dataset,
            batch_size=config["data"]["bs"],
            shuffle=True,
        )
    elif config["eval"]["data"] == "test":
        loader = dataloader_type(
            test_dataset,
            batch_size=config["data"]["bs"],
            shuffle=False,
        )
    else:
        raise ValueError("Invalid data split.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = MODEL_LIST[config["model"]["type"]]
    model = model_type(config).to(device)

    # Load checkpoint
    checkpoint_path = f"{CHECKPOINT_PATH}/{config['model']['name']}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])

    eval_type = EVAL_LIST[config["data"]["type"]]
    evaluator = eval_type(
        model,
        f"{LOG_PATH}/{config["model"]["name"]}.csv",
        device,
        save=config["eval"]["save"],
    )

    if config["eval"]["evaluate"]:
        evaluator.evaluate(loader)

    if config["eval"]["static_predict"]:
        evaluator.static_predict(config["eval"]["image_path"])

    if config["eval"]["plot_history"]:
        evaluator.plot_history()

    if config["eval"]["realtime_predict"]:
        evaluator.realtime_predict()

    if config["eval"]["confusion_matrix"]:
        evaluator.confusion_matrix(loader)
