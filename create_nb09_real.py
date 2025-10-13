#!/usr/bin/env python3
import json

cells = []
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Model Evaluation - Data Science Koans\n",
        "\n",
        "Master model evaluation techniques!\n",
        "\n",
        "## What You Will Learn\n",
        "- Cross-validation\n",
        "- ROC curves and AUC\n",
        "- Learning curves\n",
        "- Validation strategies\n",
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
        "from sklearn.model_selection import cross_val_score, KFold\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import roc_curve, roc_auc_score\n",
        "from koans.core.validator import KoanValidator\n",
        "from koans.core.progress import ProgressTracker\n",
        "\n",
        "validator = KoanValidator('09_model_evaluation')\n",
        "tracker = ProgressTracker()\n",
        "print('Setup complete!')\n",
        "print(f\"Progress: {tracker.get_notebook_progress('09_model_evaluation')}%\")"
    ]
})

koans = [
    ("9.1: Cross Validation", "K-fold CV", "Intermediate",
     "def do_cv():\n    from sklearn.model_selection import cross_val_score\n    from sklearn.linear_model import LogisticRegression\n    X = np.array([[1], [2], [3], [4], [5], [6]])\n    y = np.array([0, 0, 0, 1, 1, 1])\n    model = LogisticRegression()\n    # TODO: Perform 3-fold cross-validation\n    # Use cross_val_score(model, X, y, cv=3)\n    pass",
     "scores = do_cv()\nassert len(scores) == 3\nassert isinstance(scores, np.ndarray)"),
    
    ("9.2: CV Mean Score", "Average performance", "Intermediate",
     "def cv_mean():\n    scores = np.array([0.8, 0.9, 0.85])\n    # TODO: Return mean of scores\n    pass",
     "result = cv_mean()\nassert result == 0.85"),
    
    ("9.3: KFold Splitter", "Manual CV splits", "Intermediate",
     "def create_kfold():\n    from sklearn.model_selection import KFold\n    # TODO: Create KFold with n_splits=5\n    pass",
     "kf = create_kfold()\nassert kf.n_splits == 5"),
    
    ("9.4: Train-Val-Test Split", "3-way split", "Intermediate",
     "def split_3way():\n    from sklearn.model_selection import train_test_split\n    X = np.arange(100).reshape(-1, 1)\n    y = np.arange(100)\n    # TODO: Split into 60% train, 20% val, 20% test\n    # First split 80/20, then split the 80% into 75/25\n    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)\n    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)\n    return X_train, X_val, X_test",
     "tr, val, te = split_3way()\nassert len(tr) == 60\nassert len(val) == 20\nassert len(te) == 20"),
    
    ("9.5: ROC Curve", "TPR vs FPR", "Intermediate",
     "def compute_roc():\n    from sklearn.metrics import roc_curve\n    y_true = np.array([0, 0, 1, 1])\n    y_scores = np.array([0.1, 0.4, 0.6, 0.9])\n    # TODO: Compute ROC curve\n    # Use roc_curve(y_true, y_scores)\n    pass",
     "fpr, tpr, thresholds = compute_roc()\nassert len(fpr) > 0\nassert len(tpr) > 0"),
    
    ("9.6: AUC Score", "Area under ROC", "Intermediate",
     "def compute_auc():\n    from sklearn.metrics import roc_auc_score\n    y_true = np.array([0, 0, 1, 1])\n    y_scores = np.array([0.1, 0.4, 0.6, 0.9])\n    # TODO: Compute AUC score\n    pass",
     "score = compute_auc()\nassert 0 <= score <= 1\nassert score > 0.5"),
    
    ("9.7: Stratified Split", "Preserve class ratios", "Intermediate",
     "def stratified_split():\n    from sklearn.model_selection import StratifiedKFold\n    # TODO: Create StratifiedKFold with 3 splits\n    pass",
     "skf = stratified_split()\nassert skf.n_splits == 3"),
    
    ("9.8: Validation Curve Concept", "Param vs score", "Intermediate",
     "def validation_concept():\n    # In a validation curve, we vary a hyperparameter\n    # and measure train/validation scores\n    # TODO: Return True if you understand this\n    pass",
     "result = validation_concept()\nassert result == True"),
    
    ("9.9: Overfitting Detection", "Train vs test gap", "Intermediate",
     "def detect_overfit():\n    train_score = 0.99\n    test_score = 0.65\n    # TODO: Return True if gap > 0.2 (overfitting indicator)\n    pass",
     "result = detect_overfit()\nassert result == True"),
    
    ("9.10: Cross-Val with Shuffle", "Randomized splits", "Intermediate",
     "def cv_shuffle():\n    from sklearn.model_selection import KFold\n    # TODO: Create KFold with 5 splits and shuffle=True\n    pass",
     "kf = cv_shuffle()\nassert kf.n_splits == 5\nassert kf.shuffle == True")
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
    "source": ["## Congratulations!\n\nYou completed Model Evaluation!"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "progress = tracker.get_notebook_progress('09_model_evaluation')\n",
        "print(f'Final Progress: {progress}%')\n",
        "if progress == 100:\n",
        "    print('Excellent! You mastered Model Evaluation!')"
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

with open('koans/notebooks/09_model_evaluation.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook 09 created successfully!")