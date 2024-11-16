from typing import Callable, Union, List, Optional, Dict
import signal
import time
from statistics import mean
import math
import concurrent.futures
import multiprocessing
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt
from utils.input_generator import generate_random_valid_input, generate_random_invalid_input, INPUT_SIZES

# Constants
TIMEOUT_SECONDS = 5  # Maximum time allowed for each test case
TOTAL_TIMEOUT_SECONDS = 60  # Maximum time allowed for all tests combined

SearchFunction = Callable[
    [List[Union[int, str]], Union[int, str]], Optional[int]]

# Context manager for timing out function execution


@contextmanager
def timeout_context(seconds):
    """Context manager for timing out function execution."""
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")

    # Set up the signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)


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


def analyze_complexity(input_size: int, time_taken: float) -> Dict[str, bool]:
    """Analyze if the time complexity matches various complexity patterns."""
    if input_size <= 1 or time_taken <= 0:
        return {
            'matches_log_n': False,
            'matches_linear': False,
            'matches_nlogn': False,
            'matches_quadratic': False,
            'matches_exponential': False,
            'matches_factorial': False
        }

    try:
        # Base complexity values using natural log for better numerical stability
        n = float(input_size)
        log_n = math.log2(n)
        n_log_n = n * log_n
        n_squared = n * n

        # Calculate normalized growth rates
        # We use a smaller scaling factor and natural log for better stability
        scale = 1e3  # Convert to milliseconds
        time_scaled = time_taken * scale

        # For logarithmic complexity, we expect the ratio to grow very slowly
        log_ratio = time_scaled / log_n
        linear_ratio = time_scaled / n
        nlogn_ratio = time_scaled / n_log_n
        quadratic_ratio = time_scaled / n_squared

        # Only calculate these for small inputs to avoid overflow
        if input_size <= 20:
            exp_ratio = time_scaled / (2.0 ** n)
            factorial_ratio = time_scaled / math.factorial(input_size)
        else:
            exp_ratio = float('inf')
            factorial_ratio = float('inf')

    except (OverflowError, ValueError, ZeroDivisionError):
        return {
            'matches_log_n': False,
            'matches_linear': False,
            'matches_nlogn': False,
            'matches_quadratic': False,
            'matches_exponential': False,
            'matches_factorial': False
        }

    # Get all valid ratios
    ratios = {
        'log_n': log_ratio,
        'linear': linear_ratio,
        'nlogn': nlogn_ratio,
        'quadratic': quadratic_ratio,
        'exp': exp_ratio,
        'factorial': factorial_ratio
    }

    # Filter out invalid ratios
    valid_ratios = {k: v for k, v in ratios.items()
                    if v != float('inf') and not math.isnan(v) and v > 0}

    if not valid_ratios:
        return {
            'matches_log_n': False,
            'matches_linear': False,
            'matches_nlogn': False,
            'matches_quadratic': False,
            'matches_exponential': False,
            'matches_factorial': False
        }

    # Special handling for logarithmic complexity
    # For log(n), the ratio should grow very slowly compared to other complexities
    min_ratio = min(valid_ratios.values())

    # Use different thresholds for different complexities
    log_threshold = min_ratio * 2.0  # More lenient for logarithmic growth
    linear_threshold = min_ratio * 1.5
    nlogn_threshold = min_ratio * 1.5
    quadratic_threshold = min_ratio * 1.5
    exp_threshold = min_ratio * 1.5
    factorial_threshold = min_ratio * 1.5

    # A complexity matches if its ratio is within its threshold of the minimum
    return {
        'matches_log_n': log_ratio <= log_threshold,
        'matches_linear': linear_ratio <= linear_threshold,
        'matches_nlogn': nlogn_ratio <= nlogn_threshold,
        'matches_quadratic': quadratic_ratio <= quadratic_threshold,
        'matches_exponential': exp_ratio <= exp_threshold,
        'matches_factorial': factorial_ratio <= factorial_threshold
    }


def format_complexity_matches(complexity_matches: Dict[str, bool]) -> str:
    """Format complexity matches into a readable string."""
    matches = []
    if complexity_matches['matches_log_n']:
        matches.append('O(log n)')
    if complexity_matches['matches_linear']:
        matches.append('O(n)')
    if complexity_matches['matches_nlogn']:
        matches.append('O(n log n)')
    if complexity_matches['matches_quadratic']:
        matches.append('O(n²)')
    if complexity_matches['matches_exponential']:
        matches.append('O(2^n)')
    if complexity_matches['matches_factorial']:
        matches.append('O(n!)')

    return ', '.join(matches) if matches else 'Unknown'


def run_single_test(
    search_function: SearchFunction,
    size: int,
    use_strings: bool,
    is_valid: bool,
    requires_sorted: bool = True
) -> Dict:
    """Run a single test case with the given parameters."""
    try:
        # Generate test data
        if is_valid:
            input_list, target = generate_random_valid_input(
                size, use_strings, sorted_array=requires_sorted)
        else:
            input_list, target = generate_random_invalid_input(
                size, use_strings, sorted_array=requires_sorted)

        # Measure execution time
        start_time = time.time()
        try:
            result = search_function(input_list, target)
            execution_time = time.time() - start_time

            # Check if execution took too long
            if execution_time > TIMEOUT_SECONDS:
                return {
                    'status': 'TIMEOUT',
                    'time': execution_time,
                    'size': size,
                    'error': f'Execution exceeded {TIMEOUT_SECONDS} seconds',
                    'array': input_list,
                    'target': target,
                    'result': result
                }

            # Validate result
            if is_valid and result is None:
                return {
                    'status': 'FAIL',
                    'time': execution_time,
                    'size': size,
                    'error': 'Failed to find valid target',
                    'array': input_list,
                    'target': target,
                    'result': result
                }
            if not is_valid and result is not None:
                return {
                    'status': 'FAIL',
                    'time': execution_time,
                    'size': size,
                    'error': 'Found invalid target',
                    'array': input_list,
                    'target': target,
                    'result': result
                }

            return {
                'status': 'SUCCESS',
                'time': execution_time,
                'size': size,
                'error': None,
                'array': input_list,
                'target': target,
                'result': result
            }

        except Exception as e:
            return {
                'status': 'ERROR',
                'time': time.time() - start_time,
                'size': size,
                'error': str(e),
                'array': input_list,
                'target': target,
                'result': None
            }

    except Exception as e:
        return {
            'status': 'ERROR',
            'time': None,
            'size': size,
            'error': str(e),
            'array': None,
            'target': None,
            'result': None
        }


def plot_complexity_analysis(sizes: List[int], times: List[float], algorithm_name: str = "Algorithm"):
    """Create a plot comparing actual execution times with theoretical complexity curves.

    Args:
        sizes: List of input sizes
        times: List of execution times
        algorithm_name: Name of the algorithm being analyzed
    """
    plt.figure(figsize=(12, 8))

    # Convert to numpy arrays for easier manipulation
    sizes = np.array(sizes)
    times = np.array(times)

    # Plot actual times
    plt.scatter(sizes, times, label='Actual times', color='blue', zorder=5)
    plt.plot(sizes, times, 'b--', alpha=0.5)

    # Normalize theoretical curves to match actual data scale
    max_time = np.max(times)
    scale_factor = max_time / np.max(sizes)

    # Generate theoretical curves
    log_n = np.log2(sizes) * scale_factor
    linear = sizes * scale_factor
    n_log_n = sizes * np.log2(sizes) * scale_factor
    quadratic = sizes * sizes * scale_factor

    # Plot theoretical curves
    plt.plot(sizes, log_n, 'g-', label='O(log n)', alpha=0.5)
    plt.plot(sizes, linear, 'y-', label='O(n)', alpha=0.5)
    plt.plot(sizes, n_log_n, 'r-', label='O(n log n)', alpha=0.5)
    plt.plot(sizes, quadratic, 'm-', label='O(n²)', alpha=0.5)

    # Customize plot
    plt.title(f'Time Complexity Analysis: {algorithm_name}')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Use log scales for better visualization
    plt.xscale('log')
    plt.yscale('log')

    # Show plot in a window
    plt.show(block=False)

    # Keep the window open for a while
    plt.pause(10)  # Keep window open for 10 seconds
    plt.close()


def test_search_on_random_inputs(
    search_function: SearchFunction,
    requires_sorted: bool = True
):
    """Test a search function with random inputs of various sizes."""
    print(f"Testing {search_function.__name__}...")

    # Calculate number of workers based on CPU count
    max_workers = max(1, multiprocessing.cpu_count() // 2)
    print(f"Running with {max_workers} parallel workers (CPU count / 2)")

    # Store results
    results = []
    sizes = []
    times = []

    print("\nTest Results:")
    print("-" * 120)
    print(f"{'Array Size':<12} | {'Test Type':<15} | {'Time (sec)':<12} | {
          'Status':<10} | {'Observed Complexity':<40}")
    print("-" * 120)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {}

        # Submit all test cases
        for size in INPUT_SIZES:
            # Test with valid and invalid inputs
            for is_valid in [True, False]:
                # Test with both numbers and strings
                for use_strings in [False, True]:
                    future = executor.submit(
                        run_single_test,
                        search_function,
                        size,
                        use_strings,
                        is_valid,
                        requires_sorted
                    )
                    future_to_params[future] = (size, is_valid, use_strings)

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_params):
            size, is_valid, use_strings = future_to_params[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    if result['status'] == 'SUCCESS':
                        sizes.append(size)
                        times.append(result['time'])

                    # Print result
                    test_type = f"{'Valid' if is_valid else 'Invalid'} {
                        'String' if use_strings else 'Number'}"
                    complexity = result.get('complexity', 'Unknown')
                    print(f"{size:<12} {test_type:<15} | {result['time']:<12.6f} | {
                          result['status']:<10} | {complexity:<40}")

                    if 'details' in result:
                        print(f"\nDetailed results for size {size}:")
                        print(result['details'])
                        print("-" * 50)

            except Exception as e:
                print(f"Error processing result for size {size}: {str(e)}")

    total_time = sum(result['time'] for result in results if 'time' in result)
    print(f"\nTotal test time: {total_time:.2f} seconds")

    if sizes and times:
        # Create visualization
        plot_complexity_analysis(sizes, times, search_function.__name__)
        print("\nComplexity analysis plot has been saved as 'complexity_analysis.png'")

    print("\nSummary:")
    print(f"Average execution time: {
          mean([r['time'] for r in results if 'time' in r]):.6f} seconds")
    print(f"Largest input size tested: {max(sizes) if sizes else 'N/A'}")
