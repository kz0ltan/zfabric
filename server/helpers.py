"""Helper functions for zfabric"""

import random


def load_file(path, default=None):
    """Load a file or: raise an exception/return default"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
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
