import argparse
from pathlib import Path


def parse_config_path_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
    )
    return parser.parse_args()