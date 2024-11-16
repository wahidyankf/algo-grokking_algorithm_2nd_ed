# Algo: Grokking Algorithm 2nd Edition

This repository contains my code implementations and notes while working through the book "Grokking Algorithms: An Illustrated Guide for Programmers and Other Curious People (2nd Edition)" by Aditya Bhargava.

## Features

- Visual Complexity Analysis

  - Real-time plotting of algorithm performance
  - Comparison with theoretical complexity curves (O(log n), O(n), O(n log n), O(n²))
  - Interactive visualization using matplotlib
  - Logarithmic scaling for better pattern visibility

- Comprehensive Testing Framework

  - Parallel test execution
  - Automatic complexity detection
  - Support for both numeric and string inputs
  - Configurable timeout settings
  - Detailed performance reporting

- Search Algorithms
  - Binary Search (O(log n))
  - Simple Search (O(n))
  - Type-safe implementations
  - Support for both numbers and strings

## Project Structure

```
algo_grokking_algorithm_2nd_ed/
├── ch01_introduction_to_algorithms/
│   ├── 01a_simple_search.py   # Simple linear search implementation
│   └── 01b_binary_search.py   # Binary search implementation
└── utils/
    ├── input_generator.py     # Test data generation utilities
    └── test_utils.py         # Testing and analysis framework
```

## Setup

1. Make sure you have Python 3.12+ installed
2. Install Poetry (dependency management tool)
3. Clone this repository
4. Run `poetry install` to install dependencies
5. Run `poetry shell` to activate the virtual environment

## Running the Code

### Running Search Algorithm Tests

To run a specific search algorithm with visualization:

```bash
poetry run python -m algo_grokking_algorithm_2nd_ed.ch01_introduction_to_algorithms.01b_binary_search
```

This will:

1. Run the algorithm against various input sizes
2. Display a real-time plot of performance characteristics
3. Compare actual performance with theoretical complexity curves
4. Show detailed test results in the console

## Dependencies

- Python 3.12+
- Poetry for dependency management
- Key packages:
  - matplotlib: For complexity visualization
  - numpy: For numerical computations
  - pytest: For testing (coming soon)

## Features in Development

- [ ] More search algorithm implementations
- [ ] Additional visualization options
- [ ] Comprehensive test suite
- [ ] Performance benchmarking tools
- [ ] Interactive algorithm demonstrations

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements or find any bugs.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
