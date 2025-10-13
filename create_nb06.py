#!/usr/bin/env python3
import json

cells = []
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Feature Engineering - Data Science Koans\n",
        "\n",
        "Master feature engineering!\n",
        "\n",
        "## What You Will Learn\n",
        "- Date/time features\n",
        "- Text features\n",
        "- Aggregations\n",
        "- Lag and rolling features\n",
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
        "validator = KoanValidator('06_feature_engineering')\n",
        "tracker = ProgressTracker()\n",
        "print('Setup complete!')\n",
        "print(f\"Progress: {tracker.get_notebook_progress('06_feature_engineering')}%\")"
    ]
})

koans = [
    ("6.1: Date Parts", "Extract components", "Intermediate",
     "def extract_date():\n    df = pd.DataFrame({'date': pd.date_range('2023-01-01', periods=3)})\n    # TODO: Add year, month columns\n    pass",
     "result = extract_date()\nassert 'year' in result.columns"),
    
    ("6.2: Time Features", "Hour/minute", "Intermediate",
     "def time_features():\n    df = pd.DataFrame({'time': pd.date_range('2023-01-01', periods=24, freq='H')})\n    # TODO: Add hour column\n    pass",
     "result = time_features()\nassert 'hour' in result.columns"),
    
    ("6.3: Text Length", "String metrics", "Intermediate",
     "def text_len():\n    df = pd.DataFrame({'text': ['hi', 'hello', 'world']})\n    # TODO: Add length column\n    pass",
     "result = text_len()\nassert result['length'].iloc[1] == 5"),
    
    ("6.4: Word Count", "Count words", "Intermediate",
     "def word_cnt():\n    df = pd.DataFrame({'text': ['one', 'one two']})\n    # TODO: Add word_count\n    pass",
     "result = word_cnt()\nassert result['word_count'].iloc[1] == 2"),
    
    ("6.5: Ratio Features", "Compute ratios", "Intermediate",
     "def make_ratio():\n    df = pd.DataFrame({'a': [10, 20], 'b': [5, 10]})\n    # TODO: Add ratio = a / b\n    pass",
     "result = make_ratio()\nassert result['ratio'].iloc[0] == 2.0"),
    
    ("6.6: Group Mean", "Aggregation", "Intermediate",
     "def group_agg():\n    df = pd.DataFrame({'cat': ['A','A','B'], 'val': [10,20,30]})\n    # TODO: Add cat_mean\n    pass",
     "result = group_agg()\nassert 'cat_mean' in result.columns"),
    
    ("6.7: Lag Features", "Previous value", "Intermediate",
     "def make_lag():\n    df = pd.DataFrame({'val': [1, 2, 3]})\n    # TODO: Add lag_1\n    pass",
     "result = make_lag()\nassert pd.isna(result['lag_1'].iloc[0])"),
    
    ("6.8: Rolling Mean", "Moving average", "Intermediate",
     "def rolling_avg():\n    df = pd.DataFrame({'val': [1, 2, 3, 4, 5]})\n    # TODO: Add roll_2 (window=2)\n    pass",
     "result = rolling_avg()\nassert 'roll_2' in result.columns"),
    
    ("6.9: Frequency Encode", "Count encoding", "Intermediate",
     "def freq_encode():\n    df = pd.DataFrame({'cat': ['A','B','A']})\n    # TODO: Add cat_freq\n    pass",
     "result = freq_encode()\nassert result['cat_freq'].iloc[0] == 2"),
    
    ("6.10: Target Encode", "Mean encoding", "Intermediate",
     "def target_encode():\n    df = pd.DataFrame({'cat': ['A','A','B'], 'y': [1,2,3]})\n    # TODO: Add cat_mean_y\n    pass",
     "result = target_encode()\nassert 'cat_mean_y' in result.columns")
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
    "source": ["## Congratulations!\n\nYou completed Feature Engineering!"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "progress = tracker.get_notebook_progress('06_feature_engineering')\n",
        "print(f'Final Progress: {progress}%')\n",
        "if progress == 100:\n",
        "    print('Excellent! You mastered Feature Engineering!')"
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

with open('koans/notebooks/06_feature_engineering.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook 06 created successfully!")