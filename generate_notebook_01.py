#!/usr/bin/env python3
"""Generate complete Notebook 01 with all 10 koans"""

import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
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

# Helper function to create cells
def md_cell(content):
    return {"cell_type": "markdown", "metadata": {}, "source": content.split("\n")}

def code_cell(content):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": content.split("\n")}

# Add cells
notebook["cells"].extend([
    md_cell("""# NumPy Fundamentals - Data Science Koans

Welcome to your first Data Science Koan!

## What You Will Learn
- Creating NumPy arrays
- Understanding array properties
- Array indexing and slicing
- Array operations and broadcasting
- Essential array methods

## How to Use
1. Read each koan carefully
2. Complete TODO sections
3. Run validation
4. Iterate until passing"""),

    code_cell("""# Setup - Run first!
import sys
sys.path.append('../..')
import numpy as np
from koans.core.validator import KoanValidator
from koans.core.progress import ProgressTracker

validator = KoanValidator("01_numpy_fundamentals")
tracker = ProgressTracker()
print("Setup complete!")
print(f"Progress: {tracker.get_notebook_progress('01_numpy_fundamentals')}%")"""),

    md_cell("""## KOAN 1.1: Array Creation
**Objective**: Create arrays from lists
**Difficulty**: Beginner"""),

    code_cell("""def create_simple_array():
    # TODO: Return np.array([1, 2, 3, 4, 5])
    pass

@validator.koan(1, "Array Creation", difficulty="Beginner")
def validate():
    result = create_simple_array()
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert np.array_equal(result, np.array([1, 2, 3, 4, 5]))
validate()"""),

    md_cell("""## KOAN 1.2: Multi-dimensional Arrays
**Objective**: Create 2D arrays
**Difficulty**: Beginner"""),

    code_cell("""def create_matrix():
    # TODO: Create 3x3 matrix with values 1-9
    pass

@validator.koan(2, "Multi-dimensional Arrays", difficulty="Beginner")
def validate():
    result = create_matrix()
    expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    assert np.array_equal(result, expected)
validate()"""),

    md_cell("""## KOAN 1.3: Array Properties
**Objective**: Understand shape, dtype, ndim, size
**Difficulty**: Beginner"""),

    code_cell("""def create_zeros_array():
    # TODO: Create 2D array shape (3,4) with zeros
    pass

@validator.koan(3, "Array Properties", difficulty="Beginner")
def validate():
    result = create_zeros_array()
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 4)
    assert result.ndim == 2
    assert result.size == 12
    assert np.all(result == 0)
validate()"""),

    md_cell("""## KOAN 1.4: Array Creation Functions
**Objective**: Use np.arange, np.linspace, etc
**Difficulty**: Beginner"""),

    code_cell("""def create_range_array():
    # TODO: Create array [0, 2, 4, 6, 8] using np.arange
    pass

@validator.koan(4, "Array Creation Functions", difficulty="Beginner")
def validate():
    result = create_range_array()
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([0, 2, 4, 6, 8]))
validate()"""),

    md_cell("""## KOAN 1.5: Array Indexing
**Objective**: Access single elements
**Difficulty**: Beginner"""),

    code_cell("""def get_last_element():
    arr = np.array([10, 20, 30, 40, 50])
    # TODO: Return the last element (50)
    pass

@validator.koan(5, "Array Indexing", difficulty="Beginner")
def validate():
    result = get_last_element()
    assert result == 50
validate()"""),

    md_cell("""## KOAN 1.6: Array Slicing
**Objective**: Extract subarrays
**Difficulty**: Beginner"""),

    code_cell("""def slice_middle_elements():
    arr = np.array([10, 20, 30, 40, 50])
    # TODO: Return middle 3 elements [20, 30, 40]
    pass

@validator.koan(6, "Array Slicing", difficulty="Beginner")
def validate():
    result = slice_middle_elements()
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([20, 30, 40]))
validate()"""),

    md_cell("""## KOAN 1.7: 2D Indexing and Slicing
**Objective**: Access matrix elements
**Difficulty**: Beginner"""),

    code_cell("""def get_second_row():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # TODO: Return the second row [4, 5, 6]
    pass

@validator.koan(7, "2D Indexing", difficulty="Beginner")
def validate():
    result = get_second_row()
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([4, 5, 6]))
validate()"""),

    md_cell("""## KOAN 1.8: Array Operations
**Objective**: Element-wise arithmetic
**Difficulty**: Beginner"""),

    code_cell("""def multiply_array():
    arr = np.array([1, 2, 3, 4, 5])
    # TODO: Multiply all elements by 2
    pass

@validator.koan(8, "Array Operations", difficulty="Beginner")
def validate():
    result = multiply_array()
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([2, 4, 6, 8, 10]))
validate()"""),

    md_cell("""## KOAN 1.9: Broadcasting
**Objective**: Operations between different shapes
**Difficulty**: Beginner"""),

    code_cell("""def add_to_each_row():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    row = np.array([10, 20, 30])
    # TODO: Add row to each row of matrix