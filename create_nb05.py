#!/usr/bin/env python3
import json

cells = []
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Data Transformation - Data Science Koans\n",
        "\n",
        "Master data transformation techniques!\n",
        "\n",
        "## What You Will Learn\n",
        "- Scaling and normalization\n",
        "- Encoding categorical variables\n",
        "- Binning and discretization\n",
        "- Log transforms\n",
        "- Feature combinations\n",
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
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n",
        "from koans.core.validator import KoanValidator\n",
        "from koans.core.progress import ProgressTracker\n",
        "\n",
        "validator = KoanValidator('05_data_transformation')\n",
        "tracker = ProgressTracker()\n",
        "print('Setup complete!')\n",
        "print(f\"Progress: {tracker.get_notebook_progress('05_data_transformation')}%\")"
    ]
})

koans = [
    ("5.1: Min-Max Scaling", "Scale to 0-1", "Beginner",
     "def min_max_scale():\n    data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)\n    # TODO: Scale using MinMaxScaler\n    pass",
     "result = min_max_scale()\nassert result[0][0] == 0.0\nassert result[-1][0] == 1.0"),
    
    ("5.2: Standard Scaling", "Z-score normalization", "Beginner",
     "def standard_scale():\n    data = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)\n    # TODO: Use StandardScaler\n    pass",
     "result = standard_scale()\nassert abs(result.mean()) < 0.01\nassert abs(result.std() - 1.0) < 0.01"),
    
    ("5.3: One-Hot Encoding", "Convert categories", "Beginner",
     "def one_hot_encode():\n    df = pd.DataFrame({'color': ['red', 'blue', 'red', 'green']})\n    # TODO: Use pd.get_dummies\n    pass",
     "result = one_hot_encode()\nassert result.shape[1] == 3"),
    
    ("5.4: Label Encoding", "Numeric categories", "Beginner",
     "def label_encode():\n    categories = ['low', 'medium', 'high', 'low', 'high']\n    # TODO: Map to 0, 1, 2\n    pass",
     "result = label_encode()\nassert len(set(result)) == 3"),
    
    ("5.5: Binning", "Discretize continuous", "Beginner",
     "def bin_ages():\n    ages = [5, 15, 25, 35, 45, 55, 65]\n    # TODO: Create bins: child, adult, senior\n    pass",
     "result = bin_ages()\nassert len(set(result)) == 3"),
    
    ("5.6: Log Transform", "Handle skewness", "Beginner",
     "def log_transform():\n    data = np.array([1, 10, 100, 1000])\n    # TODO: Apply np.log10\n    pass",
     "result = log_transform()\nassert result[-1] == 3.0"),
    
    ("5.7: Power Transform", "Square/sqrt", "Beginner",
     "def square_root():\n    data = np.array([1, 4, 9, 16, 25])\n    # TODO: Apply square root\n    pass",
     "result = square_root()\nassert result[-1] == 5.0"),
    
    ("5.8: Interaction Features", "Combine features", "Beginner",
     "def create_interaction():\n    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})\n    # TODO: Create column C = A * B\n    pass",
     "result = create_interaction()\nassert 'C' in result.columns\nassert result['C'].iloc[0] == 4"),
    
    ("5.9: Polynomial Features", "Higher order terms", "Beginner",
     "def add_squared():\n    df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})\n    # TODO: Add column x_squared\n    pass",
     "result = add_squared()\nassert 'x_squared' in result.columns\nassert result['x_squared'].iloc[-1] == 25"),
    
    ("5.10: Boolean Flags", "Create indicators", "Beginner",
     "def create_flag():\n    df = pd.DataFrame({'value': [5, 15, 25, 35]})\n    # TODO: Add is_high flag (value > 20)\n    pass",
     "result = create_flag()\nassert 'is_high' in result.columns\nassert result['is_high'].sum() == 2")
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
    "source": ["## Congratulations!\n\nYou completed Data Transformation!"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "progress = tracker.get_notebook_progress('05_data_transformation')\n",
        "print(f'Final Progress: {progress}%')\n",
        "if progress == 100:\n",
        "    print('Excellent! You mastered Data Transformation!')"
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

with open('koans/notebooks/05_data_transformation.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook 05 created successfully!")