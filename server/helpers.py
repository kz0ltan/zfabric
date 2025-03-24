"""Helper functions for zfabric"""

import argparse
import os
import random


def load_file(path, default=None, type: str = "r"):
    """Load a file or: raise an exception/return default"""
    try:
        with open(path, type, encoding="utf-8" if type == "r" else None) as fp:
            return fp.read()
    except FileNotFoundError as e:
        if default is None:
            raise e
        return default


def generate_random_number(length: int) -> int:
    """Generate a random number with given length"""
    if length <= 0:
        raise ValueError("Length must be a positive integer")

    # Calculate the minimum and maximum values for the given length
    min_value = 10 ** (length - 1)
    max_value = 10**length - 1

    # Generate a random number within the specified range
    return random.randint(min_value, max_value)


def ensure_directories_exist(path: str) -> None:
    """Checks if the directories in the provided path exist and creates them if not"""
    abs_path = os.path.expanduser(os.path.dirname(path))

    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
