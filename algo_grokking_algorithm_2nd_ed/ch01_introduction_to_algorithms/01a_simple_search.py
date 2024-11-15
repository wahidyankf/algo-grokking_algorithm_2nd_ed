import sys
import os
from typing import List, Union, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This function implements a linear search algorithm, which is a simple
# algorithm that searches through the array one element at a time until it
# finds the target value or it has searched the entire array.


def simple_search(array: List[Union[int, str]], target: Union[int, str]) -> Optional[int]:
    """Linear search through array to find target

    Args:
        array: List of items to search through (all numbers or all strings)
        target: Item to find in the array (same type as array elements)

    Returns:
        Index of target if found, None if not found
    """
    # Iterate through the array
    for i in range(len(array)):
        # If the current element in the array is equal to the target value,
        # return the index of the element
        if array[i] == target:
            return i
    # If the target value is not found, return None
    return None


if __name__ == "__main__":
    # Import a function from the test_utils module that tests the search
    # function with random inputs.
    from ch01_introduction_to_algorithms.utils.test_utils import test_search_on_random_inputs
    # Call the test_search_on_random_inputs function with the simple_search
    # function as an argument.
    test_search_on_random_inputs(simple_search)
