from typing import Callable, Union, List, Optional, Dict
import signal
import time
from statistics import mean
from utils.input_generator import generate_random_valid_input, generate_random_invalid_input, INPUT_SIZES

SearchFunction = Callable[[
    List[Union[int, str]], Union[int, str]], Optional[int]]


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Test took too long to complete")


def run_with_timeout(func: Callable, timeout: int = 15):
    """Run a function with a timeout

    Args:
        func: Function to run
        timeout: Timeout in seconds

    Returns:
        Function result if completed within timeout

    Raises:
        TimeoutError if function takes too long
    """
    # Set the signal handler and a timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        result = func()
    finally:
        # Disable the alarm
        signal.alarm(0)
    return result


def print_performance_summary(performance_data: Dict[int, Dict[str, float]]):
    """Print a summary table of algorithm performance

    Args:
        performance_data: Dictionary mapping input size to timing data
    """
    print("\nPerformance Summary")
    print("=" * 100)
    print(f"{'Input Size':>12} | {'Avg Time (s)':>12} | {
          'Time Ratio':>12} | {'Expected O(n)':>12} | {'Match O(n)?':>12}")
    print("-" * 100)

    prev_time = None
    prev_size = None

    for size in sorted(performance_data.keys()):
        avg_time = performance_data[size].get('avg_time', float('inf'))
        if avg_time == float('inf'):
            print(f"{size:>12} | {'TIMEOUT':>12} | {
                  'N/A':>12} | {'N/A':>12} | {'N/A':>12}")
            continue

        # Calculate ratios
        if prev_time is not None and prev_size is not None:
            time_ratio = avg_time / prev_time
            size_ratio = size / prev_size
            expected_ratio = size_ratio  # For O(n)
            # Allow some variance
            is_linear = 0.5 <= (time_ratio / size_ratio) <= 2.0

            print(f"{size:>12} | {avg_time:>12.6f} | {time_ratio:>12.2f} | {
                  expected_ratio:>12.2f} | {'✓' if is_linear else '✗':>12}")
        else:
            print(f"{size:>12} | {avg_time:>12.6f} | {
                  'N/A':>12} | {'N/A':>12} | {'N/A':>12}")

        prev_time = avg_time
        prev_size = size

    print("=" * 100)
    print("Time Ratio: Actual time increase between consecutive input sizes")
    print("Expected O(n): Expected time increase for linear O(n) complexity")
    print("Match O(n)?: ✓ if actual ratio is within 50-200% of expected ratio")


def test_search_on_random_inputs(search_function: SearchFunction, num_valid_tests: int = 2, num_invalid_tests: int = 1) -> None:
    """Generic test function for search algorithms

    Args:
        search_function: The search function to test
        num_valid_tests: Number of tests with valid inputs (target exists)
        num_invalid_tests: Number of tests with invalid inputs (target doesn't exist)
    """
    # Store performance data
    number_performance: Dict[int, Dict[str, float]] = {}
    string_performance: Dict[int, Dict[str, float]] = {}

    # Test with random number inputs
    print(f"\nTesting {search_function.__name__} with number inputs:")
    print("=" * 50)

    for size in INPUT_SIZES:
        print(f"\nTesting with array size: {size}")
        print("-" * 30)

        test_times = []
        try:
            print("\nValid number inputs:")
            print("-" * 20)
            for i in range(num_valid_tests):
                start_time = time.time()

                def test_func():
                    (array, target) = generate_random_valid_input(
                        size=size, use_strings=False)
                    result = search_function(array, target)
                    elapsed = time.time() - start_time
                    test_times.append(elapsed)
                    print(f"Test {i + 1}:")
                    if size <= 100:  # Only print full array for small sizes
                        print(f"Array: {array}")
                    else:
                        print(f"Array: [{array[0]}, {array[1]}, ..., {
                              array[-2]}, {array[-1]}]")
                    print(f"Target: {target}")
                    print(f"Found at index: {result}")
                    print(f"Time taken: {elapsed:.2f} seconds")
                    print("-" * 50)
                    return result

                run_with_timeout(test_func)

            print("\nInvalid number inputs:")
            print("-" * 20)
            for i in range(num_invalid_tests):
                start_time = time.time()

                def test_func():
                    (array, target) = generate_random_invalid_input(
                        size=size, use_strings=False)
                    result = search_function(array, target)
                    elapsed = time.time() - start_time
                    test_times.append(elapsed)
                    print(f"Test {i + 1}:")
                    if size <= 100:  # Only print full array for small sizes
                        print(f"Array: {array}")
                    else:
                        print(f"Array: [{array[0]}, {array[1]}, ..., {
                              array[-2]}, {array[-1]}]")
                    print(f"Target: {target}")
                    print(f"Found at index: {result}")
                    print(f"Time taken: {elapsed:.2f} seconds")
                    print("-" * 50)
                    return result

                run_with_timeout(test_func)

            # Store average time for this size
            number_performance[size] = {'avg_time': mean(test_times)}

        except TimeoutError:
            print(f"\n⚠️  Tests for size {
                  size} aborted - took longer than 15 seconds")
            print("Skipping remaining tests for this size")
            number_performance[size] = {'avg_time': float('inf')}
            continue

    # Test with random string inputs
    print(f"\nTesting {search_function.__name__} with string inputs:")
    print("=" * 50)

    for size in INPUT_SIZES:
        print(f"\nTesting with array size: {size}")
        print("-" * 30)

        test_times = []
        try:
            print("\nValid string inputs:")
            print("-" * 20)
            for i in range(num_valid_tests):
                start_time = time.time()

                def test_func():
                    (array, target) = generate_random_valid_input(
                        size=size, use_strings=True)
                    result = search_function(array, target)
                    elapsed = time.time() - start_time
                    test_times.append(elapsed)
                    print(f"Test {i + 1}:")
                    if size <= 100:  # Only print full array for small sizes
                        print(f"Array: {array}")
                    else:
                        print(f"Array: [{array[0]}, {array[1]}, ..., {
                              array[-2]}, {array[-1]}]")
                    print(f"Target: {target}")
                    print(f"Found at index: {result}")
                    print(f"Time taken: {elapsed:.2f} seconds")
                    print("-" * 50)
                    return result

                run_with_timeout(test_func)

            print("\nInvalid string inputs:")
            print("-" * 20)
            for i in range(num_invalid_tests):
                start_time = time.time()

                def test_func():
                    (array, target) = generate_random_invalid_input(
                        size=size, use_strings=True)
                    result = search_function(array, target)
                    elapsed = time.time() - start_time
                    test_times.append(elapsed)
                    print(f"Test {i + 1}:")
                    if size <= 100:  # Only print full array for small sizes
                        print(f"Array: {array}")
                    else:
                        print(f"Array: [{array[0]}, {array[1]}, ..., {
                              array[-2]}, {array[-1]}]")
                    print(f"Target: {target}")
                    print(f"Found at index: {result}")
                    print(f"Time taken: {elapsed:.2f} seconds")
                    print("-" * 50)
                    return result

                run_with_timeout(test_func)

            # Store average time for this size
            string_performance[size] = {'avg_time': mean(test_times)}

        except TimeoutError:
            print(f"\n⚠️  Tests for size {
                  size} aborted - took longer than 15 seconds")
            print("Skipping remaining tests for this size")
            string_performance[size] = {'avg_time': float('inf')}
            continue

    # Print performance summaries
    print("\nNumber Search Performance")
    print_performance_summary(number_performance)

    print("\nString Search Performance")
    print_performance_summary(string_performance)
