#!/usr/bin/env python3
"""Generate notebooks 08-15"""
import json

def make_notebook(num, title, nb_id, koans_list):
    cells = []
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [
        f"# {title} - Data Science Koans\n", "\n", f"Master {title.lower()}!\n", "\n",
        "## How to Use\n", "1. Read each koan\n", "2. Complete TODOs\n",
        "3. Run validation\n", "4. Iterate"]})
    
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "# Setup\n", "import sys\n", "sys.path.append('../..')\n",
        "import numpy as np\n", "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from koans.core.validator import KoanValidator\n",
        "from koans.core.progress import ProgressTracker\n", "\n",
        f"validator = KoanValidator('{nb_id}')\n",
        "tracker = ProgressTracker()\n", "print('Setup complete!')\n",
        f"print(f\"Progress: {{tracker.get_notebook_progress('{nb_id}')}}%\")"]})
    
    for i, (t, o, d, c, v) in enumerate(koans_list, 1):
        cells.append({"cell_type": "markdown", "metadata": {}, 
                     "source": [f"## KOAN {num}.{i}: {t}\n**Objective**: {o}\n**Difficulty**: {d}"]})
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": [f"{c}\n\n@validator.koan({i}, \"{t}\", difficulty=\"{d}\")\ndef validate():\n    {v}\nvalidate()"]})
    
    cells.append({"cell_type": "markdown", "metadata": {}, 
                 "source": [f"## Congratulations!\n\nYou completed {title}!"]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": [f"progress = tracker.get_notebook_progress('{nb_id}')\n",
                           "print(f'Final Progress: {progress}%')\n", "if progress == 100:\n",
                           f"    print('Excellent! You mastered {title}!')"]})
    
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py",
          "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python",
          "pygments_lexer": "ipython3", "version": "3.8.0"}}, "nbformat": 4, "nbformat_minor": 4}
    
    with open(f'koans/notebooks/{nb_id}.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Created {nb_id}.ipynb")

# NB08: Classification
nb08 = [
    ("Binary Classification", "Two classes", "Intermediate", "def binary_classify():\n    from sklearn.linear_model import LogisticRegression\n    X = [[1], [2], [3], [4]]\n    y = [0, 0, 1, 1]\n    # TODO: Fit model\n    pass", "model = binary_classify()\nassert hasattr(model, 'predict')"),
    ("Predict Class", "Get predictions", "Intermediate", "def predict_class():\n    from sklearn.linear_model import LogisticRegression\n    model = LogisticRegression()\n    model.fit([[1], [4]], [0, 1])\n    # TODO: Predict [[2]]\n    pass", "result = predict_class()\nassert result[0] in [0, 1]"),
    ("Predict Proba", "Get probabilities", "Intermediate", "def predict_proba():\n    from sklearn.linear_model import LogisticRegression\n    model = LogisticRegression()\n    model.fit([[1], [4]], [0, 1])\n    # TODO: predict_proba [[2]]\n    pass", "result = predict_proba()\nassert result.shape[1] == 2"),
    ("Accuracy Score", "Calculate accuracy", "Intermediate", "def calc_accuracy():\n    from sklearn.metrics import accuracy_score\n    y_true = [0, 1, 1, 0]\n    y_pred = [0, 1, 1, 0]\n    # TODO: Return accuracy\n    pass", "result = calc_accuracy()\nassert result == 1.0"),
    ("Confusion Matrix", "True/False pos/neg", "Intermediate", "def get_confusion():\n    from sklearn.metrics import confusion_matrix\n    y_true = [0, 1, 0, 1]\n    y_pred = [0, 1, 0, 1]\n    # TODO: Return matrix\n    pass", "result = get_confusion()\nassert result.shape == (2, 2)"),
    ("Precision", "TP / (TP + FP)", "Intermediate", "def calc_precision():\n    from sklearn.metrics import precision_score\n    y_true = [0, 1, 1, 0]\n    y_pred = [0, 1, 1, 0]\n    # TODO: Return precision\n    pass", "result = calc_precision()\nassert result == 1.0"),
    ("Recall", "TP / (TP + FN)", "Intermediate", "def calc_recall():\n    from sklearn.metrics import recall_score\n    y_true = [0, 1, 1, 0]\n    y_pred = [0, 1, 1, 0]\n    # TODO: Return recall\n    pass", "result = calc_recall()\nassert result == 1.0"),
    ("F1 Score", "Harmonic mean", "Intermediate", "def calc_f1():\n    from sklearn.metrics import f1_score\n    y_true = [0, 1, 1, 0]\n    y_pred = [0, 1, 1, 0]\n    # TODO: Return F1\n    pass", "result = calc_f1()\nassert result == 1.0"),
    ("Multi-class", "More than 2 classes", "Intermediate", "def multi_class():\n    from sklearn.linear_model import LogisticRegression\n    X = [[1], [2], [3]]\n    y = [0, 1, 2]\n    # TODO: Fit model\n    pass", "model = multi_class()\nassert hasattr(model, 'classes_')"),
    ("Decision Boundary", "Classification threshold", "Intermediate", "def classify_threshold():\n    proba = 0.7\n    # TODO: Return 1 if >= 0.5 else 0\n    pass", "result = classify_threshold()\nassert result == 1")
]

# NB09: Model Evaluation
nb09 = [
    ("Cross Validation", "K-fold CV", "Intermediate", "def cross_val():\n    from sklearn.model_selection import cross_val_score\n    from sklearn.linear_model import LogisticRegression\n    X = [[1], [2], [3], [4], [5]]\