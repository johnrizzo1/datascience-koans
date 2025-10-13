#!/usr/bin/env python3
import json

cells = []
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Data Cleaning - Data Science Koans\n",
        "\n",
        "Master data cleaning techniques!\n",
        "\n",
        "## What You Will Learn\n",
        "- Handling missing values\n",
        "- Removing duplicates\n",
        "- Data type conversions\n",
        "- String cleaning\n",
        "- Date/time parsing\n",
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
        "validator = KoanValidator('04_data_cleaning')\n",
        "tracker = ProgressTracker()\n",
        "print('Setup complete!')\n",
        "print(f\"Progress: {tracker.get_notebook_progress('04_data_cleaning')}%\")"
    ]
})

koans = [
    ("4.1: Drop Missing Values", "Remove nulls", "Beginner",
     "def drop_nulls():\n    df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})\n    # TODO: Drop rows with any null values\n    pass",
     "result = drop_nulls()\nassert len(result) == 1"),
    
    ("4.2: Fill Missing Values", "Impute nulls", "Beginner",
     "def fill_nulls():\n    df = pd.DataFrame({'A': [1, None, 3, None, 5]})\n    # TODO: Fill nulls with mean value\n    pass",
     "result = fill_nulls()\nassert result.isna().sum() == 0"),
    
    ("4.3: Remove Duplicates", "Drop dupes", "Beginner",
     "def remove_dupes():\n    df = pd.DataFrame({'A': [1, 2, 2, 3, 3, 3]})\n    # TODO: Remove duplicate rows\n    pass",
     "result = remove_dupes()\nassert len(result) == 3"),
    
    ("4.4: Convert Data Types", "Cast columns", "Beginner",
     "def convert_types():\n    df = pd.DataFrame({'nums': ['1', '2', '3']})\n    # TODO: Convert 'nums' column to int\n    pass",
     "result = convert_types()\nassert result['nums'].dtype == 'int64'"),
    
    ("4.5: Strip Whitespace", "Clean strings", "Beginner",
     "def strip_spaces():\n    df = pd.DataFrame({'text': ['  hello  ', '  world  ']})\n    # TODO: Strip whitespace from 'text' column\n    pass",
     "result = strip_spaces()\nassert result['text'].iloc[0] == 'hello'"),
    
    ("4.6: Lowercase Strings", "Normalize case", "Beginner",
     "def lower_case():\n    df = pd.DataFrame({'text': ['HELLO', 'WORLD']})\n    # TODO: Convert 'text' column to lowercase\n    pass",
     "result = lower_case()\nassert result['text'].iloc[0] == 'hello'"),
    
    ("4.7: Replace Values", "Substitute data", "Beginner",
     "def replace_vals():\n    df = pd.DataFrame({'status': ['yes', 'no', 'yes', 'no']})\n    # TODO: Replace 'yes' with 1 and 'no' with 0\n    pass",
     "result = replace_vals()\nassert result['status'].iloc[0] == 1"),
    
    ("4.8: Parse Dates", "Convert to datetime", "Beginner",
     "def parse_dates():\n    df = pd.DataFrame({'date': ['2023-01-01', '2023-01-02']})\n    # TODO: Convert 'date' column to datetime\n    pass",
     "result = parse_dates()\nassert pd.api.types.is_datetime64_any_dtype(result['date'])"),
    
    ("4.9: Handle Outliers", "Cap extremes", "Beginner",
     "def cap_outliers():\n    df = pd.DataFrame({'val': [1, 2, 3, 100, 4, 5]})\n    # TODO: Cap values above 10 to 10\n    pass",
     "result = cap_outliers()\nassert result['val'].max() == 10"),
    
    ("4.10: Rename Columns", "Fix names", "Beginner",
     "def rename_cols():\n    df = pd.DataFrame({'Old Name': [1, 2, 3]})\n    # TODO: Rename 'Old Name' to 'new_name'\n    pass",
     "result = rename_cols()\nassert 'new_name' in result.columns")
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

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Congratulations!\n\nYou completed Data Cleaning!"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "progress = tracker.get_notebook_progress('04_data_cleaning')\n",
        "print(f'Final Progress: {progress}%')\n",
        "if progress == 100:\n",
        "    print('Excellent! You mastered Data Cleaning!')"
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

with open('koans/notebooks/04_data_cleaning.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook 04 created successfully!")