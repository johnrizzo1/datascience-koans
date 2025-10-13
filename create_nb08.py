#!/usr/bin/env python3
import json

cells = []
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Classification Basics - Data Science Koans\n",
        "\n",
        "Master classification!\n",
        "\n",
        "## What You Will Learn\n",
        "- Binary and multi-class classification\n",
        "- Model evaluation metrics\n",
        "- Confusion matrices\n",
        "- Precision, recall, F1\n",
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
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score\n",
        "from koans.core.validator import KoanValidator\n",
        "from koans.core.progress import ProgressTracker\n",
        "\n",
        "validator = KoanValidator('08_classification_basics')\n",
        "tracker = ProgressTracker()\n",
        "print('Setup complete!')\n",
        "print(f\"Progress: {tracker.get_notebook_progress('08_classification_basics')}%\")"
    ]
})

koans = [
    ("8.1: Binary Classification", "Two classes", "Intermediate",
     "def binary_classify():\n    X = [[1], [2], [3], [4]]\n    y = [0, 0, 1, 1]\n    model = LogisticRegression()\n    # TODO: Fit model\n    pass",
     "model = binary_classify()\nassert hasattr(model, 'predict')"),
    
    ("8.2: Predict Class", "Get predictions", "Intermediate",
     "def predict_class():\n    model = LogisticRegression()\n    model.fit([[1], [4]], [0, 1])\n    # TODO: Predict [[2]]\n    pass",
     "result = predict_class()\nassert result[0] in [0, 1]"),
    
    ("8.3: Predict Probabilities", "Class probabilities", "Intermediate",
     "def predict_proba():\n    model = LogisticRegression()\n    model.fit([[1], [4]], [0, 1])\n    # TODO: predict_proba [[2]]\n    pass",
     "result = predict_proba()\nassert result.shape[1] == 2"),
    
    ("8.4: Accuracy Score", "Correct predictions", "Intermediate",
     "def calc_accuracy():\n    y_true = [0, 1, 1, 0]\n    y_pred = [0, 1, 1, 0]\n    # TODO: Calculate accuracy\n    pass",
     "result = calc_accuracy()\nassert result == 1.0"),
    
    ("8.5: Confusion Matrix", "TP/FP/TN/FN", "Intermediate",
     "def get_confusion():\n    from sklearn.metrics import confusion_matrix\n    y_true = [0, 1, 0, 1]\n    y_pred = [0, 1, 0, 1]\n    # TODO: Return matrix\n    pass",
     "result = get_confusion()\nassert result.shape == (2, 2)"),
    
    ("8.6: Precision", "TP/(TP+FP)", "Intermediate",
     "def calc_precision():\n    y_true = [0, 1, 1, 0]\n    y_pred = [0, 1, 1, 0]\n    # TODO: Calculate precision\n    pass",
     "result = calc_precision()\nassert result == 1.0"),
    
    ("8.7: Recall", "TP/(TP+FN)", "Intermediate",
     "def calc_recall():\n    y_true = [0, 1, 1, 0]\n    y_pred = [0, 1, 1, 0]\n    # TODO: Calculate recall\n    pass",
     "result = calc_recall()\nassert result == 1.0"),
    
    ("8.8: F1 Score", "Harmonic mean", "Intermediate",
     "def calc_f1():\n    from sklearn.metrics import f1_score\n    y_true = [0, 1, 1, 0]\n    y_pred = [0, 1, 1, 0]\n    # TODO: Calculate F1\n    pass",
     "result = calc_f1()\nassert result == 1.0"),
    
    ("8.9: Multi-class", "3+ classes", "Intermediate",
     "def multi_class():\n    X = [[1], [2], [3]]\n    y = [0, 1, 2]\n    model = LogisticRegression()\n    # TODO: Fit model\n    pass",
     "model = multi_class()\nassert hasattr(model, 'classes_')"),
    
    ("8.10: Decision Threshold", "Classification cutoff", "Intermediate",
     "def classify_threshold():\n    proba = 0.7\n    # TODO: Return 1 if >= 0.5 else 0\n    pass",
     "result = classify_threshold()\nassert result == 1")
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
    "source": ["## Congratulations!\n\nYou completed Classification Basics!"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "progress = tracker.get_notebook_progress('08_classification_basics')\n",
        "print(f'Final Progress: {progress}%')\n",
        "if progress == 100:\n",
        "    print('Excellent! You mastered Classification!')"
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

with open('koans/notebooks/08_classification_basics.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook 08 created successfully!")