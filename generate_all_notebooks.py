#!/usr/bin/env python3
"""Generate all Data Science Koans notebooks"""

import json
import os

def create_notebook(nb_num, title, koans_data):
    """Create a complete notebook with given koans"""
    cells = []
    
    # Title and intro
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {title} - Data Science Koans\n",
            "\n",
            f"Welcome to Notebook {nb_num}!\n",
            "\n",
            "## How to Use\n",
            "1. Read each koan carefully\n",
            "2. Complete TODO sections\n",
            "3. Run validation\n",
            "4. Iterate until passing\n",
            "5. Move to next koan"
        ]
    })
    
    # Setup cell
    nb_slug = title.lower().replace(' ', '_').replace('-', '_')
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
            "import pandas as pd\n",
            "from koans.core.validator import KoanValidator\n",
            "from koans.core.progress import ProgressTracker\n",
            "\n",
            f"validator = KoanValidator(\"{nb_slug}\")\n",
            "tracker = ProgressTracker()\n",
            "print(\"Setup complete!\")\n",
            f"print(f\"Progress: {{tracker.get_notebook_progress('{nb_slug}')}}%\")"
        ]
    })
    
    # Add each koan
    for i, koan in enumerate(koans_data, 1):
        title_text, objective, difficulty, code, validation = koan
        
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"## KOAN {nb_num}.{i}: {title_text}\n",
                f"**Objective**: {objective}\n",
                f"**Difficulty**: {difficulty}"
            ]
        })
        
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"{code}\n",
                "\n",
                f"@validator.koan({i}, \"{title_text}\", difficulty=\"{difficulty}\")\n",
                "def validate():\n",
                f"    {validation}\n",
                "validate()"
            ]
        })
    
    # Completion cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## Congratulations!\n\nYou completed {title}!"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f"progress = tracker.get_notebook_progress('{nb_slug}')\n",
            "print(f\"Final Progress: {progress}%\")\n",
            "if progress == 100:\n",
            f"    print(\"Excellent! You mastered {title}!\")"
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
    
    filename = f"koans/notebooks/{nb_num:02d}_{nb_slug}.ipynb"
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Created: {filename}")

# Notebook 02: Pandas Essentials
nb02_koans = [
    ("Creating Series", "Learn 1D labeled data structure",
     "Beginner",
     "def create_series():\n    # TODO: Create Series from [10, 20, 30] with index ['a', 'b', 'c']\n    pass",
     "result = create_series()\nassert isinstance(result, pd.Series)\nassert len(result) == 3\nassert list(result.index) == ['a', 'b', 'c']"),
    
    ("Series Operations", "Perform vectorized operations on Series",
     "Beginner",
     "def double_series():\n    s = pd.Series([1, 2, 3, 4, 5])\n    # TODO: Return series with all values doubled\n    pass",
     "result = double_series()\nassert isinstance(result, pd.Series)\nassert result.tolist() == [2, 4, 6, 8, 10]"),
    
    ("Creating DataFrames", "Learn 2D labeled data structure",
     "Beginner",
     "def create_dataframe():\n    # TODO: Create DataFrame with cols 'name', 'age' and 2 rows\n    pass",
     "result = create_dataframe()\nassert isinstance(result, pd.DataFrame)\nassert list(result.columns) == ['name', 'age']\nassert len(result) >= 2"),
    
    ("DataFrame Properties", "Inspect DataFrame structure",
     "Beginner",
     "def get_shape():\n    df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})\n    # TODO: Return the shape as a tuple\n    pass",
     "result = get_shape()\nassert result == (3, 2)"),
    
    ("Column Selection", "Access DataFrame columns",
     "Beginner",
     "def select_column():\n    df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6], 'C': [7,8,9]})\n    # TODO: Return column 'B' as a Series\n    pass",
     "result = select_column()\nassert isinstance(result, pd.Series)\nassert result.tolist() == [4, 5, 6]"),
    
    ("Row Selection with loc", "Use label-based indexing",
     "Beginner",
     "def select_row():\n    df = pd.DataFrame({'A': [1,2,3]}, index=['x', 'y', 'z'])\n    # TODO: Return row 'y' using loc\n    pass",
     "result = select_row()\nassert isinstance(result, pd.Series)\nassert result['A'] == 2"),
    
    ("Boolean Indexing", "Filter rows with conditions",
     "Beginner",
     "def filter_data():\n    df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})\n    # TODO: Return rows where value > 25\n    pass",
     "result = filter_data()\nassert isinstance(result, pd.DataFrame)\nassert len(result) == 3\nassert result['value'].min() > 25"),
    
    ("Basic Statistics", "Calculate summary statistics",
     "Beginner",
     "def calc_mean():\n    df = pd.DataFrame({'values': [10, 20,