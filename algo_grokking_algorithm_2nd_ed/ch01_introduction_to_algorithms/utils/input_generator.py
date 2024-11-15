from typing import Tuple, Union, List
import random
import string


def generate_random_number() -> int:
    """Generate a random number"""
    return random.randint(1, 100)


def generate_random_string() -> str:
    """Generate a random string"""
    return ''.join(random.choices(string.ascii_lowercase, k=random.randint(1, 5)))


def generate_random_valid_input(size: int = 10, use_strings: bool = False) -> Tuple[List[Union[int, str]], Union[int, str]]:
    """Generate a random array and a target that exists in the array

    Args:
        size: Size of the array to generate
        use_strings: If True, generate string array. If False, generate number array

    Returns:
        A tuple containing (array, target) where target exists in array
    """
    array: List[Union[int, str]] = []
    generator = generate_random_string if use_strings else generate_random_number

    # Generate unique items
    while len(array) < size:
        item = generator()
        if item not in array:
            array.append(item)

    # Pick a random target from the array
    target = random.choice(array)
    return array, target


def generate_random_invalid_input(size: int = 10, use_strings: bool = False) -> Tuple[List[Union[int, str]], Union[int, str]]:
    """Generate a random array and a target that doesn't exist in the array

    Args:
        size: Size of the array to generate
        use_strings: If True, generate string array. If False, generate number array

    Returns:
        A tuple containing (array, target) where target does not exist in array
    """
    array: List[Union[int, str]] = []
    generator = generate_random_string if use_strings else generate_random_number

    # Generate unique items
    while len(array) < size:
        item = generator()
        if item not in array:
            array.append(item)

    # Generate a target that's not in the array
    max_attempts = 100
    attempts = 0
    while attempts < max_attempts:
        target = generator()
        if target not in array:
            return array, target
        attempts += 1

    # If we couldn't find a unique target after max attempts,
    # modify an existing item slightly
    if use_strings:
        target = array[0] + 'x'
    else:
        target = max(array) + 1
    return array, target
