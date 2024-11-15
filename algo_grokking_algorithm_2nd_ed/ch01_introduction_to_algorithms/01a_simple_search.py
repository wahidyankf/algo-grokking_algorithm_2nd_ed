import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simple search is an algorithm that finds the position of a target value
# within an array. If an element is present in the array, then it returns
# the position of the element. Otherwise, it returns null.


def simple_search(array, target):
    """Linear search through array to find target"""
    for i in range(len(array)):
        if array[i] == target:
            return i
    return None


if __name__ == "__main__":
    from ch01_introduction_to_algorithms.utils.test_utils import test_search_on_random_inputs
    test_search_on_random_inputs(simple_search)
