#!/usr/bin/env python3
"""Generate Notebook 02: Pandas Essentials"""
import json

cells = []

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Pandas Essentials - Data Science Koans\n",
        "\n",
        "Master pandas fundamentals!\n",
        "\n",
        "## What You Will Learn\n",
        "- Creating Series and DataFrames\n",
        "- Selecting and filtering data\n",
        "- Basic statistics\n",
        "- GroupBy operations\n",
        "\n",
        "## How to Use\n",
        "1. Read each koan\n",
        "2. Complete TODOs\n",
        "3. Run validation\n",
        "4. Iterate"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Setup\n",
        "import sys\n",
        "sys.path.append('../..')\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from koans.core.validator import KoanValidator\n",
        "from koans.core.progress import ProgressTracker\n",
        "\n",
        "validator = KoanValidator('02_pandas_essentials')\n",
        "tracker = ProgressTracker()\n",
        "print('Setup complete!')\n",
        "print(f\"Progress: {tracker.get_notebook_progress('02_pandas_essentials')}%\")"
    ]
})

# Koans
koans = [
    ("Creating Series", "Learn 1D labeled data", "Beginner",
     ["def create_series():",
      "    # TODO: Create Series from [10, 20, 30] with index ['a', 'b', 'c']",
      "    pass"],
     ["result = create_series()",
      "assert isinstance(result, pd.Series)",
      "assert len(result) == 3",
      "assert list(result.index) == ['a', 'b', 'c']"]),
    
    ("Series Operations", "Vectorized ops on Series", "Beginner",
     ["def double_series():",
      "    s = pd.Series([1, 2, 3, 4, 5])",
      "    # TODO: Return series with all values doubled",
      "    pass"],
     ["result = double_series()",
      "assert isinstance(result, pd.Series)",
      "assert result.tolist() == [2, 4, 6, 8, 10]"]),
    
    ("Creating DataFrames", "Learn 2D labeled data", "Beginner",
     ["def create_dataframe():",
      "    # TODO: Create DataFrame with cols 'name', 'age'",
      "    # Use data: [('Alice', 25), ('Bob', 30)]",
      "    pass"],
     ["result = create_dataframe()",
      "assert isinstance(result, pd.DataFrame)",
      "assert list(result.columns) == ['name', 'age']",
      "assert len(result) == 2"]),
    
    ("DataFrame Properties", "Inspect structure", "Beginner",
     ["def get_shape():",
      "    df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})",
      "    # TODO: Return the shape",
      "    pass"],
     ["result = get_shape()",
      "assert result == (3, 2)"]),
    
    ("Column Selection", "Access columns", "Beginner",
     ["def select_column():",
      "    df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6], 'C': [7,8,9]})",
      "    # TODO: Return column 'B'",
      "    pass"],
     ["result = select_column()",
      "assert isinstance(result, pd.Series)",
      "assert result.tolist() == [4, 5, 6]"]),
    
    ("Row Selection with loc", "Label-based indexing", "Beginner",
     ["def select_row():",
      "    df = pd.DataFrame({'A': [1,2,3]}, index=['x', 'y', 'z'])",
      "    # TODO: Return row 'y' using loc",
      "    pass"],
     ["result = select_row()",
      "assert isinstance(result, pd.Series)",
      "assert result['A'] == 2"]),
    
    ("Boolean Indexing", "Filter with conditions", "Beginner",
     ["def filter_data():",
      "    df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})",
      "    # TODO: Return rows where value > 25",
      "    pass"],
     ["result = filter_data()",
      "assert isinstance(result, pd.DataFrame)",
      "assert len(result) == 3",
      "assert result['value'].min() > 25"]),
    
    ("Basic Statistics", "Calculate summaries", "Beginner",
     ["def calc_mean():",
      "    df = pd.DataFrame({'values': [10, 20, 30, 40, 50]})",
      "    # TODO: Return mean of 'values' column",
      "    pass"],
     ["result = calc_mean()",
      "assert result == 30.0"]),
    
    ("GroupBy Operations", "Split-apply-combine", "Beginner",
     ["def group_and_sum():",
      "    df = pd.DataFrame({",
      "        'category': ['A', 'B', 'A', 'B'],",
      "        'values': [10, 20, 30, 40]",
      "    })",
      "    # TODO: Group by 'category' and sum 'values'",
      "    pass"],
     ["result = group_and_sum()",
      "assert isinstance(result, pd.Series)",
      "assert result['A'] == 40",
      "assert result['B'] == 60"]),
    
    ("Sorting Data", "Order by values", "Beginner",
     ["def sort_by_column():",
      "    df = pd.DataFrame({'values': [30, 10, 20]})",
      "    # TODO: Sort by 'values' ascending",
      "    pass"],
     ["result = sort_by_column()",
      "assert isinstance(result, pd.DataFrame)",
      "assert result['values'].tolist() == [10, 20, 30]"])
]

for i, (title, obj, diff, code_lines, val_lines) in enumerate(koans, 1):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## KOAN 2.{i}: {title}\n**Objective**: {obj}\n**Difficulty**: {diff}"]
    })
    
    source = code_lines + [""] + [f"@validator.koan({i}, '{title}', difficulty='{diff}')"] + ["def validate():"] + ["    " + line for line in val_lines] + ["validate()"]
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    })

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Congratulations!\n\nYou completed Pandas Essentials!"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "progress = tracker.get_notebook_progress('02_pandas_essentials')\n",
        "print(f'Final Progress: {progress}%')\n",
        "if progress == 100:\n",
        "    print('Excellent! You mastered Pandas Essentials!')"
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

with open('koans/notebooks/02_pandas_essentials.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook 02 created successfully!")