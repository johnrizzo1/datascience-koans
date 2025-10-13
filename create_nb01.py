#!/usr/bin/env python3
import json

# Build complete notebook with all 10 koans
cells = []

# Title
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# NumPy Fundamentals - Data Science Koans\n",
        "\n",
        "Welcome to your first Data Science Koan!\n",
        "\n",
        "## What You Will Learn\n",
        "- Creating NumPy arrays\n",
        "- Understanding array properties\n",
        "- Array indexing and slicing\n",
        "- Array operations and broadcasting\n",
        "- Essential array methods\n",
        "\n",
        "## How to Use\n",
        "1. Read each koan carefully\n",
        "2. Complete TODO sections\n",
        "3. Run validation\n",
        "4. Iterate until passing"
    ]
})

# Setup
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Setup - Run first!\n",
        "import sys\n",
        "sys.path.append('../..')\n",
        "import numpy as np\n",
        "from koans.core.validator import KoanValidator\n",
        "from koans.core.progress import ProgressTracker\n",
        "\n",
        "validator = KoanValidator(\"01_numpy_fundamentals\")\n",
        "tracker = ProgressTracker()\n",
        "print(\"Setup complete!\")\n",
        "print(f\"Progress: {tracker.get_notebook_progress('01_numpy_fundamentals')}%\")"
    ]
})

# Koans 1-10
koans = [
    ("1.1: Array Creation", "Create arrays from lists", "Beginner", 
     "def create_simple_array():\n    # TODO: Return np.array([1, 2, 3, 4, 5])\n    pass",
     "result = create_simple_array()\nassert isinstance(result, np.ndarray)\nassert result.shape == (5,)\nassert np.array_equal(result, np.array([1, 2, 3, 4, 5]))"),
    
    ("1.2: Multi-dimensional Arrays", "Create 2D arrays", "Beginner",
     "def create_matrix():\n    # TODO: Create 3x3 matrix with values 1-9\n    pass",
     "result = create_matrix()\nexpected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\nassert isinstance(result, np.ndarray)\nassert result.shape == (3, 3)\nassert np.array_equal(result, expected)"),
    
    ("1.3: Array Properties", "Understand shape, dtype, ndim, size", "Beginner",
     "def create_zeros_array():\n    # TODO: Create 2D array shape (3,4) with zeros\n    pass",
     "result = create_zeros_array()\nassert isinstance(result, np.ndarray)\nassert result.shape == (3, 4)\nassert result.ndim == 2\nassert result.size == 12\nassert np.all(result == 0)"),
    
    ("1.4: Array Creation Functions", "Use np.arange", "Beginner",
     "def create_range_array():\n    # TODO: Create array [0, 2, 4, 6, 8] using np.arange\n    pass",
     "result = create_range_array()\nassert isinstance(result, np.ndarray)\nassert np.array_equal(result, np.array([0, 2, 4, 6, 8]))"),
    
    ("1.5: Array Indexing", "Access single elements", "Beginner",
     "def get_last_element():\n    arr = np.array([10, 20, 30, 40, 50])\n    # TODO: Return the last element (50)\n    pass",
     "result = get_last_element()\nassert result == 50"),
    
    ("1.6: Array Slicing", "Extract subarrays", "Beginner",
     "def slice_middle_elements():\n    arr = np.array([10, 20, 30, 40, 50])\n    # TODO: Return middle 3 elements [20, 30, 40]\n    pass",
     "result = slice_middle_elements()\nassert isinstance(result, np.ndarray)\nassert np.array_equal(result, np.array([20, 30, 40]))"),
    
    ("1.7: 2D Indexing", "Access matrix elements", "Beginner",
     "def get_second_row():\n    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n    # TODO: Return the second row [4, 5, 6]\n    pass",
     "result = get_second_row()\nassert isinstance(result, np.ndarray)\nassert np.array_equal(result, np.array([4, 5, 6]))"),
    
    ("1.8: Array Operations", "Element-wise arithmetic", "Beginner",
     "def multiply_array():\n    arr = np.array([1, 2, 3, 4, 5])\n    # TODO: Multiply all elements by 2\n    pass",
     "result = multiply_array()\nassert isinstance(result, np.ndarray)\nassert np.array_equal(result, np.array([2, 4, 6, 8, 10]))"),
    
    ("1.9: Broadcasting", "Operations between different shapes", "Beginner",
     "def add_to_each_row():\n    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n    row = np.array([10, 20, 30])\n    # TODO: Add row to each row of matrix\n    pass",
     "result = add_to_each_row()\nexpected = np.array([[11, 22, 33], [14, 25, 36], [17, 28, 39]])\nassert isinstance(result, np.ndarray)\nassert np.array_equal(result, expected)"),
    
    ("1.10: Array Methods", "Use aggregation methods", "Beginner",
     "def calculate_mean():\n    arr = np.array([10, 20, 30, 40, 50])\n    # TODO: Return the mean of the array\n    pass",
     "result = calculate_mean()\nassert result == 30.0")
]

for i, (title, obj, diff, code, val) in enumerate(koans, 1):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## KOAN {title}\n**Objective**: {obj}\n**Difficulty**: {diff}"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [f"{code}\n\n@validator.koan({i}, \"{title.split(': ')[1]}\", difficulty=\"{diff}\")\ndef validate():\n    {val}\nvalidate()"]
    })

# Final cell
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Congratulations!\n\nYou completed NumPy Fundamentals!"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "progress = tracker.get_notebook_progress('01_numpy_fundamentals')\n",
        "print(f\"Final Progress: {progress}%\")\n",
        "if progress == 100:\n",
        "    print(\"Excellent! You mastered NumPy fundamentals!\")"
    ]
})

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('koans/notebooks/01_numpy_fundamentals.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook created successfully!")