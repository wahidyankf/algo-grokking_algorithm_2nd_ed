from typing import Callable, Union, List, Optional, Dict
import signal
import time
from statistics import mean
import math
import concurrent.futures
import multiprocessing
from contextlib import contextmanager
from utils.input_generator import generate_random_valid_input, generate_random_invalid_input, INPUT_SIZES

# Constants
TIMEOUT_SECONDS = 15  # Maximum time allowed for each test case

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
    # For comparison, we'll normalize the times to a scale of 0-1
    # We'll use the input size and time to check if it follows various complexity patterns

    # O(log n)
    log_n = math.log2(input_size)
    log_ratio = time_taken / log_n if log_n > 0 else float('inf')

    # O(n)
    linear_ratio = time_taken / input_size

    # O(n log n)
    nlogn = input_size * math.log2(input_size)
    nlogn_ratio = time_taken / nlogn if nlogn > 0 else float('inf')

    # O(n²)
    try:
        quadratic = float(input_size ** 2)
        quadratic_ratio = time_taken / quadratic
    except OverflowError:
        quadratic_ratio = float('inf')

    # O(2^n)
    try:
        # For large n, we'll use log to avoid overflow
        if input_size > 100:
            exp_ratio = float('inf')
        else:
            exp = float(2 ** input_size)
            exp_ratio = time_taken / exp
    except OverflowError:
        exp_ratio = float('inf')

    # O(n!)
    try:
        # For large n, we'll use log to avoid overflow
        if input_size > 100:
            factorial_ratio = float('inf')
        else:
            factorial = float(math.factorial(input_size))
            factorial_ratio = time_taken / factorial
    except (OverflowError, ValueError):
        factorial_ratio = float('inf')

    # Compare ratios to determine the closest match
    ratios = {
        'O(log n)': log_ratio,
        'O(n)': linear_ratio,
        'O(n log n)': nlogn_ratio,
        'O(n²)': quadratic_ratio,
        'O(2^n)': exp_ratio,
        'O(n!)': factorial_ratio
    }

    # Find the most consistent ratio across different input sizes
    valid_ratios = [r for r in ratios.values() if r != float('inf')
                    and not math.isnan(r)]
    if not valid_ratios:
        return {
            'matches_log_n': False,
            'matches_linear': False,
            'matches_nlogn': False,
            'matches_quadratic': False,
            'matches_exponential': False,
            'matches_factorial': False
        }

    min_ratio = min(valid_ratios)
    threshold = min_ratio * 10  # Allow some variance

    return {
        'matches_log_n': log_ratio <= threshold and log_ratio != float('inf'),
        'matches_linear': linear_ratio <= threshold and linear_ratio != float('inf'),
        'matches_nlogn': nlogn_ratio <= threshold and nlogn_ratio != float('inf'),
        'matches_quadratic': quadratic_ratio <= threshold and quadratic_ratio != float('inf'),
        'matches_exponential': exp_ratio <= threshold and exp_ratio != float('inf'),
        'matches_factorial': factorial_ratio <= threshold and factorial_ratio != float('inf')
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


def test_search_on_random_inputs(
    search_function: SearchFunction,
    requires_sorted: bool = True
) -> None:
    """Test a search function with random inputs of various sizes."""
    # Calculate optimal number of worker threads (CPU count / 2)
    max_workers = max(1, multiprocessing.cpu_count() // 2)

    print(f"\nTesting {search_function.__name__}...")
    print(f"Running with {max_workers} parallel workers (CPU count / 2)")
    print("\nResults:")
    print(f"{'Input Size':<12} | {'Time (s)':<10} | {
          'Status':<10} | {'Complexity Matches':<30}")
    print("-" * 65)

    # Store results for complexity analysis
    results = []
    timeout_occurred = False

    # Create test cases for both numbers and strings
    test_cases = []
    for size in INPUT_SIZES:
        if not timeout_occurred:
            # Add number test cases
            test_cases.extend([
                (size, False, True),  # Valid number inputs
                (size, False, False)  # Invalid number inputs
            ])
            # Add string test cases
            test_cases.extend([
                (size, True, True),   # Valid string inputs
                (size, True, False)   # Invalid string inputs
            ])

    # Run tests in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_test = {
            executor.submit(
                run_single_test,
                search_function,
                size,
                use_strings,
                is_valid,
                requires_sorted
            ): (size, use_strings, is_valid)
            for size, use_strings, is_valid in test_cases
        }

        for future in concurrent.futures.as_completed(future_to_test):
            size, use_strings, is_valid = future_to_test[future]
            try:
                with timeout_context(TIMEOUT_SECONDS):
                    test_result = future.result()

                    # Process results
                    execution_time = test_result['time']
                    if not use_strings:  # Only analyze complexity for number tests
                        complexity_matches = analyze_complexity(
                            size, execution_time)
                        complexity_str = format_complexity_matches(
                            complexity_matches)
                        results.append((size, execution_time))
                        print(f"{size:<12} | {execution_time:.<10.6f} | {
                              test_result['status']:<10} | {complexity_str:<30}")

                    # Print detailed test results for small arrays
                    if size <= 100:
                        print(f"\nDetailed results for size {size}:")
                        print(f"Array: {test_result.get('array', None)}")
                        print(f"Target: {test_result.get('target', None)}")
                        print(f"Found at index: {
                              test_result.get('result', None)}")
                        print(f"Time taken: {execution_time:.6f} seconds")
                        print("-" * 50)

            except TimeoutError:
                print(f"{size:<12} | {'>':<10} | {
                      'TIMEOUT':<10} | {'N/A':<30}")
                timeout_occurred = True
                break
            except Exception as e:
                print(f"{size:<12} | {'N/A':<10} | {'ERROR':<10} | {'N/A':<30}")
                print(f"Error: {str(e)}")

    if results:
        avg_time = mean(time for _, time in results)
        print("\nSummary:")
        print(f"Average execution time: {avg_time:.6f} seconds")
        print(f"Largest input size tested: {results[-1][0]}")
        if timeout_occurred:
            print("Note: Testing stopped due to timeout on larger inputs")

    # Analyze complexity
    print("\nComplexity Analysis:")
    complexity_counts = {
        'O(log n)': 0,
        'O(n)': 0,
        'O(n log n)': 0,
        'O(n²)': 0,
        'O(2^n)': 0,
        'O(n!)': 0
    }

    total_samples = 0
    for size, execution_time in results:
        complexity_matches = analyze_complexity(size, execution_time)
        complexity_str = format_complexity_matches(complexity_matches)
        print(f"Input size {size}: {complexity_str}")

        # Count matches
        if complexity_matches['matches_log_n']:
            complexity_counts['O(log n)'] += 1
        if complexity_matches['matches_linear']:
            complexity_counts['O(n)'] += 1
        if complexity_matches['matches_nlogn']:
            complexity_counts['O(n log n)'] += 1
        if complexity_matches['matches_quadratic']:
            complexity_counts['O(n²)'] += 1
        if complexity_matches['matches_exponential']:
            complexity_counts['O(2^n)'] += 1
        if complexity_matches['matches_factorial']:
            complexity_counts['O(n!)'] += 1
        total_samples += 1

    # Determine the most likely complexity
    if total_samples > 0:
        print("\nComplexity Conclusion:")
        print("-" * 50)
        print("Matches per complexity class:")
        for complexity, count in complexity_counts.items():
            percentage = (count / total_samples) * 100
            print(f"{complexity}: {
                  count}/{total_samples} samples ({percentage:.1f}%)")

        # Find the complexity with highest match percentage
        most_likely = max(complexity_counts.items(), key=lambda x: x[1])
        if most_likely[1] > 0:
            match_percentage = (most_likely[1] / total_samples) * 100
            print(f"\nMost likely time complexity: {
                  most_likely[0]} ({match_percentage:.1f}% of samples)")
        else:
            print("\nUnable to determine time complexity - no clear pattern matched")
