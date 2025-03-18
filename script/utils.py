import argparse
import json


def read_command():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--character", type=str, required=True, help="Character to capture"
    )
    parser.add_argument("--delay", type=int, default=0, help="Delay after each sample")
    parser.add_argument(
        "--num_samples", type=int, default=200, help="Number of samples to capture"
    )

    args = parser.parse_args()

    return args


def read_config():
    parser = argparse.ArgumentParser(description="Load configuration file")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    # Read and parse json
    with open(args.config, "r") as f:
        config = json.load(f)

    return config
