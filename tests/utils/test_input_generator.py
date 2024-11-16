import pytest
from algo_grokking_algorithm_2nd_ed.utils.input_generator import (
    generate_random_number,
    generate_random_string,
    generate_random_valid_input,
    generate_random_invalid_input,
    INPUT_SIZES
)


def test_generate_random_number():
    """Test random number generation with various ranges."""
    # Test default range
    num = generate_random_number()
    assert 1 <= num <= 100

    # Test custom range
    min_val, max_val = 500, 1000
    num = generate_random_number(min_val, max_val)
    assert min_val <= num <= max_val

    # Test single value range
    num = generate_random_number(42, 42)
    assert num == 42


def test_generate_random_string():
    """Test random string generation with various lengths."""
    # Test default length
    string = generate_random_string()
    assert len(string) == 5
    assert string.islower()
    assert string.isalpha()

    # Test custom length
    length = 10
    string = generate_random_string(length)
    assert len(string) == length
    assert string.islower()
    assert string.isalpha()

    # Test empty string
    string = generate_random_string(0)
    assert string == ""


def test_generate_random_valid_input():
    """Test valid input generation with various configurations."""
    # Test numeric array (unsorted)
    size = 100
    array, target = generate_random_valid_input(
        size, use_strings=False, sorted_array=False)
    assert len(array) == size
    assert all(isinstance(x, int) for x in array)
    assert target in array

    # Test string array (sorted)
    array, target = generate_random_valid_input(
        size, use_strings=True, sorted_array=True)
    assert len(array) == size
    assert all(isinstance(x, str) for x in array)
    assert target in array
    assert array == sorted(array)

    # Test size adjustment
    invalid_size = 42
    array, target = generate_random_valid_input(invalid_size)
    assert len(array) == min(INPUT_SIZES, key=lambda x: abs(x - invalid_size))


def test_generate_random_invalid_input():
    """Test invalid input generation with various configurations."""
    # Test numeric array (unsorted)
    size = 100
    array, target = generate_random_invalid_input(
        size, use_strings=False, sorted_array=False)
    assert len(array) == size
    assert all(isinstance(x, int) for x in array)
    assert target not in array

    # Test string array (sorted)
    array, target = generate_random_invalid_input(
        size, use_strings=True, sorted_array=True)
    assert len(array) == size
    assert all(isinstance(x, str) for x in array)
    assert target not in array
    assert array == sorted(array)

    # Test size adjustment
    invalid_size = 42
    array, target = generate_random_invalid_input(invalid_size)
    assert len(array) == min(INPUT_SIZES, key=lambda x: abs(x - invalid_size))


def test_input_sizes_validity():
    """Test that INPUT_SIZES is properly configured."""
    assert len(INPUT_SIZES) > 0
    assert all(isinstance(x, int) for x in INPUT_SIZES)
    assert all(x > 0 for x in INPUT_SIZES)
    assert sorted(INPUT_SIZES) == INPUT_SIZES  # Should be in ascending order


def test_edge_cases():
    """Test edge cases and potential error conditions."""
    # Test minimum size
    array, target = generate_random_valid_input(1)
    assert len(array) == 1
    assert target in array

    # Test with very large size
    large_size = INPUT_SIZES[-1]
    array, target = generate_random_valid_input(large_size)
    assert len(array) == large_size
    assert target in array

    # Test invalid input with minimum size
    array, target = generate_random_invalid_input(1)
    assert len(array) == 1
    assert target not in array

    # Test that sorted arrays maintain order
    array, _ = generate_random_valid_input(100, sorted_array=True)
    assert array == sorted(array)
