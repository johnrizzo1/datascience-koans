#!/usr/bin/env python3
"""Generate all remaining notebooks (06-15) at once"""
import json

def create_notebook(nb_num, title, koans_data):
    """Create a single notebook with given koans"""
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {title} - Data Science Koans\n",
            "\n",
            f"Master {title.lower()}!\n",
            "\n",
            "## How to Use\n",
            "1. Read each koan\n",
            "2. Complete TODOs\n",
            "3. Run validation\n",
            "4. Iterate"
        ]
    })
    
    # Setup cell
    nb_id = f"{nb_num:02d}_{title.lower().replace(' ', '_')}"
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
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.model_selection import train_test_split\n",
            "from koans.core.validator import KoanValidator\n",
            "from koans.core.progress import ProgressTracker\n",
            "\n",
            f"validator = KoanValidator('{nb_id}')\n",
            "tracker = ProgressTracker()\n",
            "print('Setup complete!')\n",
            f"print(f\"Progress: {{tracker.get_notebook_progress('{nb_id}')}}%\")"
        ]
    })
    
    # Koan cells
    for i, (koan_title, obj, diff, code, val) in enumerate(koans_data, 1):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## KOAN {nb_num}.{i}: {koan_title}\n**Objective**: {obj}\n**Difficulty**: {diff}"]
        })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [f"{code}\n\n@validator.koan({i}, \"{koan_title}\", difficulty=\"{diff}\")\ndef validate():\n    {val}\nvalidate()"]
        })
    
    # Completion cells
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
            f"progress = tracker.get_notebook_progress('{nb_id}')\n",
            "print(f'Final Progress: {progress}%')\n",
            "if progress == 100:\n",
            f"    print('Excellent! You mastered {title}!')"
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
    
    filename = f'koans/notebooks/{nb_id}.ipynb'
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Created {filename}")

# Notebook 06: Feature Engineering
nb06_koans = [
    ("Extract Date Parts", "Break down dates", "Intermediate",
     "def extract_date_parts():\n    df = pd.DataFrame({'date': pd.date_range('2023-01-01', periods=5)})\n    # TODO: Add year, month, day columns\n    pass",
     "result = extract_date_parts()\nassert 'year' in result.columns\nassert 'month' in result.columns"),
    
    ("Create Time Features", "Time-based features", "Intermediate",
     "def time_features():\n    df = pd.DataFrame({'timestamp': pd.date_range('2023-01-01', periods=24, freq='H')})\n    # TODO: Add hour, is_night (hour >= 22 or hour < 6)\n    pass",
     "result = time_features()\nassert 'hour' in result.columns\nassert 'is_night' in result.columns"),
    
    ("Text Length Feature", "String metrics", "Intermediate",
     "def text_length():\n    df = pd.DataFrame({'text': ['hi', 'hello', 'hello world']})\n    # TODO: Add text_length column\n    pass",
     "result = text_length()\nassert result['text_length'].iloc[-1] == 11"),
    
    ("Word Count Feature", "Count words", "Intermediate",
     "def word_count():\n    df = pd.DataFrame({'text': ['one', 'one two', 'one two three']})\n    # TODO: Add word_count column\n    pass",
     "result = word_count()\nassert result['word_count'].iloc[-1] == 3"),
    
    ("Ratio Features", "Create ratios", "Intermediate",
     "def create_ratio():\n    df = pd.DataFrame({'sales': [100, 200, 300], 'costs': [50, 80, 120]})\n    # TODO: Add profit_margin = (sales - costs) / sales\n    pass",
     "result = create_ratio()\nassert 'profit_margin' in result.columns"),
    
    ("Aggregation Features", "Group statistics", "Intermediate",
     "def group_mean():\n    df = pd.DataFrame({'category': ['A','A','B','B'], 'value': [10,20,30,40]})\n    # TODO: Add category_mean column\n    pass",
     "result = group_mean()\nassert 'category_mean' in result.columns"),
    
    ("Lag Features", "Previous values", "Intermediate",
     "def create_lag():\n    df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})\n    # TODO: Add lag_1 column (previous value)\n    pass",
     "result = create_lag()\nassert pd.isna(result['lag_1'].iloc[0])"),
    
    ("Rolling Features", "Moving averages", "Intermediate",
     "def rolling_mean():\n    df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})\n    # TODO: Add rolling_mean_3 (window=3)\n    pass",
     "result = rolling_mean()\nassert 'rolling_mean_3' in result.columns"),
    
    ("Frequency Encoding", "Count occurrences", "Intermediate",
     "def frequency_encode():\n    df = pd.DataFrame({'category': ['A','B','A','C','A','B']})\n    # TODO: Add category_freq column\n    pass",
     "result = frequency_encode()\nassert