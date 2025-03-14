import argparse


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
