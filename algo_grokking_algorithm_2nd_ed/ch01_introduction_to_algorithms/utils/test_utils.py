from typing import Callable, Union, List, Optional
from ch01_introduction_to_algorithms.utils.input_generator import generate_random_valid_input, generate_random_invalid_input

SearchFunction = Callable[[
    List[Union[int, str]], Union[int, str]], Optional[int]]


def test_search_on_random_inputs(search_function: SearchFunction, num_valid_tests: int = 5, num_invalid_tests: int = 2) -> None:
    """Generic test function for search algorithms

    Args:
        search_function: The search function to test
        num_valid_tests: Number of tests with valid inputs (target exists)
        num_invalid_tests: Number of tests with invalid inputs (target doesn't exist)
    """
    # Test with random number inputs
    print(f"\nTesting {search_function.__name__} with number inputs:")
    print("=" * 50)

    print("\nValid number inputs:")
    print("-" * 20)
    for i in range(num_valid_tests):
        (array, target) = generate_random_valid_input(use_strings=False)
        result = search_function(array, target)
        print(f"Test {i + 1}:")
        print(f"Array: {array}")
        print(f"Target: {target}")
        print(f"Found at index: {result}")
        print("-" * 50)

    print("\nInvalid number inputs:")
    print("-" * 20)
    for i in range(num_invalid_tests):
        (array, target) = generate_random_invalid_input(use_strings=False)
        result = search_function(array, target)
        print(f"Test {i + 1}:")
        print(f"Array: {array}")
        print(f"Target: {target}")
        print(f"Found at index: {result}")
        print("-" * 50)

    # Test with random string inputs
    print(f"\nTesting {search_function.__name__} with string inputs:")
    print("=" * 50)

    print("\nValid string inputs:")
    print("-" * 20)
    for i in range(num_valid_tests):
        (array, target) = generate_random_valid_input(use_strings=True)
        result = search_function(array, target)
        print(f"Test {i + 1}:")
        print(f"Array: {array}")
        print(f"Target: {target}")
        print(f"Found at index: {result}")
        print("-" * 50)

    print("\nInvalid string inputs:")
    print("-" * 20)
    for i in range(num_invalid_tests):
        (array, target) = generate_random_invalid_input(use_strings=True)
        result = search_function(array, target)
        print(f"Test {i + 1}:")
        print(f"Array: {array}")
        print(f"Target: {target}")
        print(f"Found at index: {result}")
        print("-" * 50)
