from typing import List, Union, Optional


def binary_search(array: List[Union[int, str]], target: Union[int, str]) -> Optional[int]:
    """Binary search implementation that works with both numbers and strings.

    This function performs a binary search on a sorted array to find the target value.
    It returns the index of the target if found, otherwise returns None.
    Time complexity: O(log n) where n is the length of the array.
    Space complexity: O(1) as it only uses a constant amount of extra space.

    Args:
        array: A sorted list of integers or strings to search through
        target: The value to search for in the array

    Returns:
        Optional[int]: The index of the target if found, None otherwise

    Example:
        >>> binary_search([1, 3, 5, 7, 9], 3)
        1
        >>> binary_search(['apple', 'banana', 'orange'], 'banana')
        1
        >>> binary_search([1, 2, 3, 4, 5], 6)
        None
    """
    # Initialize the left and right pointers
    left = 0
    right = len(array) - 1

    # Continue searching while left pointer is less than or equal to right pointer
    while left <= right:
        # Calculate the middle index
        mid = (left + right) // 2

        # If target is found at mid, return the index
        if array[mid] == target:
            return mid

        # If target is less than mid value, search left half
        elif target < array[mid]:
            right = mid - 1

        # If target is greater than mid value, search right half
        else:
            left = mid + 1

    # If target is not found, return None
    return None


if __name__ == "__main__":
    # Import test utilities
    from utils.test_utils import test_search_on_random_inputs

    # Test the binary search implementation
    test_search_on_random_inputs(binary_search)
