import random
import string
from typing import List, Tuple, Union

# Available input sizes for testing algorithm performance
INPUT_SIZES = [1, 10, 100, 1000, 10000, 100000,
               1000000, 10000000]


def generate_random_number(min_val: int = 1, max_val: int = 100) -> int:
    """Generate a random number within the specified range

    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)

    Returns:
        Random integer between min_val and max_val
    """
    return random.randint(min_val, max_val)


def generate_random_string(length: int = 5) -> str:
    """Generate a random string of specified length

    Args:
        length: Length of string to generate

    Returns:
        Random string of lowercase letters
    """
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def generate_random_valid_input(
    size: int = 10,
    use_strings: bool = False,
    sorted_array: bool = False
) -> Tuple[List[Union[int, str]], Union[int, str]]:
    """Generate a random array and a target that exists in the array

    Args:
        size: Size of array to generate (defaults to 10)
        use_strings: If True, generate string array. If False, generate number array
        sorted_array: If True, sort the array before returning

    Returns:
        Tuple of (array, target) where target exists in array. Array will be sorted if sorted_array is True
    """
    if size not in INPUT_SIZES:
        size = min(INPUT_SIZES, key=lambda x: abs(x - size))

    # Generate array
    if use_strings:
        array = [generate_random_string() for _ in range(size)]
    else:
        array = [generate_random_number() for _ in range(size)]

    # Sort if requested
    if sorted_array:
        array.sort()

    # Choose a random existing target
    target = random.choice(array)

    return array, target


def generate_random_invalid_input(
    size: int = 10,
    use_strings: bool = False,
    sorted_array: bool = False
) -> Tuple[List[Union[int, str]], Union[int, str]]:
    """Generate a random array and a target that does not exist in the array

    Args:
        size: Size of array to generate (defaults to 10)
        use_strings: If True, generate string array. If False, generate number array
        sorted_array: If True, sort the array before returning

    Returns:
        Tuple of (array, target) where target does not exist in array. Array will be sorted if sorted_array is True
    """
    if size not in INPUT_SIZES:
        size = min(INPUT_SIZES, key=lambda x: abs(x - size))

    # Generate array
    if use_strings:
        array = [generate_random_string() for _ in range(size)]
        # Generate a string that's guaranteed to not be in the array
        target = 'z' * (len(max(array, key=len)) + 1)
    else:
        array = [generate_random_number() for _ in range(size)]
        # Generate a number that's guaranteed to not be in the array
        target = max(array) + 1

    # Sort if requested
    if sorted_array:
        array.sort()

    return array, target
