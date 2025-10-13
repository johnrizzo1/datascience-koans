#!/usr/bin/env python3
import json

cells = []
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Data Exploration - Data Science Koans\n",
        "\n",
        "Master exploratory data analysis!\n",
        "\n",
        "## What You Will Learn\n",
        "- Loading and profiling datasets\n",
        "- Detecting missing values\n",
        "- Data type conversions\n",
        "- Correlation analysis\n",
        "- Outlier detection\n",
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
        "import matplotlib.pyplot as plt\n",
        "from koans.core.validator import KoanValidator\n",
        "from koans.core.progress import ProgressTracker\n",
        "\n",
        "validator = KoanValidator('03_data_exploration')\n",
        "tracker = ProgressTracker()\n",
        "print('Setup complete!')\n",
        "print(f\"Progress: {tracker.get_notebook_progress('03_data_exploration')}%\")"
    ]
})

koans = [
    ("Loading CSV Data", "Read files", "Beginner",
     ["def load_csv():", "    # TODO: Create sample DataFrame", "    return pd.DataFrame({'A': [1,2,3]})"],
     ["result = load_csv()", "assert isinstance(result, pd.DataFrame)"]),
    
    ("Data Profiling", "Understand structure", "Beginner",
     ["def get_info():", "    df = pd.DataFrame({'A': [1,2,3], 'B': ['x','y','z']})", "    # TODO: Return tuple (num_rows, num_cols)", "    pass"],
     ["result = get_info()", "assert result == (3, 2)"]),
    
    ("Missing Value Detection", "Find nulls", "Beginner",
     ["def count_missing():", "    df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})", "    # TODO: Return total count of missing values", "    pass"],
     ["result = count_missing()", "assert result == 2"]),
    
    ("Data Type Analysis", "Check dtypes", "Beginner",
     ["def analyze_types():", "    df = pd.DataFrame({'nums': [1,2,3], 'text': ['a','b','c']})", "    # TODO: Return number of numeric columns", "    pass"],
     ["result = analyze_types()", "assert result == 1"]),
    
    ("Basic Visualization", "Create plots", "Beginner",
     ["def create_histogram():", "    data = [1, 2, 2, 3, 3, 3, 4, 4, 5]", "    # TODO: Create histogram and return True", "    return True"],
     ["result = create_histogram()", "assert result == True"]),
    
    ("Correlation Analysis", "Calculate correlations", "Beginner",
     ["def calc_correlation():", "    df = pd.DataFrame({'A': [1,2,3,4,5], 'B': [2,4,6,8,10]})", "    # TODO: Return correlation between A and B", "    pass"],
     ["result = calc_correlation()", "assert result == 1.0"]),
    
    ("Unique Values", "Count distinct", "Beginner",
     ["def count_unique():", "    df = pd.DataFrame({'category': ['A','B','A','C','B','A']})", "    # TODO: Return number of unique values in 'category'", "    pass"],
     ["result = count_unique()", "assert result == 3"]),
    
    ("Cross-tabulation", "Frequency tables", "Beginner",
     ["def create_crosstab():", "    df = pd.DataFrame({'X': ['A','A','B','B'], 'Y': [1,2,1,2]})", "    # TODO: Create crosstab of X and Y", "    return pd.crosstab(df['X'], df['Y'])"],
     ["result = create_crosstab()", "assert isinstance(result, pd.DataFrame)"]),
    
    ("Outlier Detection", "Find extremes", "Beginner",
     ["def detect_outliers():", "    data = [10, 12, 13, 12, 11, 100, 13]", "    # TODO: Return list of indices where value > mean + 2*std", "    pass"],
     ["result = detect_outliers()", "assert 5 in result"]),
    
    ("Data Quality Report", "Comprehensive check", "Beginner",
     ["def quality_report():", "    df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, 6]})", "    # TODO: Return dict with 'total_rows', 'total_nulls'", "    pass"],
     ["result = quality_report()", "assert 'total_rows' in result", "assert result['total_nulls'] == 1"])
]

for i, (title, obj, diff, code_lines, val_lines) in enumerate(koans, 1):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## KOAN 3.{i}: {title}\n**Objective**: {obj}\n**Difficulty**: {diff}"]
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
    "source": ["## Congratulations!\n\nYou completed Data Exploration!"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "progress = tracker.get_notebook_progress('03_data_exploration')\n",
        "print(f'Final Progress: {progress}%')\n",
        "if progress == 100:\n",
        "    print('Excellent! You mastered Data Exploration!')"
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

with open('koans/notebooks/03_data_exploration.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook 03 created successfully!")
