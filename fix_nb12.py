#!/usr/bin/env python3
import json

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.8.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": "# Notebook 12: Ensemble Methods\n\nMaster Random Forests, Gradient Boosting, and ensemble techniques."})
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "import sys\nsys.path.append('../..')\nimport numpy as np\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.datasets import make_classification\nfrom koans.core.validator import KoanValidator\nfrom koans.core.progress import ProgressTracker\nvalidator = KoanValidator('12_ensemble_methods')\ntracker = ProgressTracker()\nprint('Setup complete!')"})

# Koans 1-7
koans = [
    ("12.1: Random Forest", "rf = RandomForestClassifier(n_estimators=100, random_state=42)", "score > 0.5"),
    ("12.2: Feature Importance", "importances = rf.feature_importances_", "len(importances) == 10"),
    ("12.3: Gradient Boosting", "gb = GradientBoostingClassifier(n_estimators=100, random_state=42)", "score > 0.5"),
    ("12.4: XGBoost", "Use XGBoost or GradientBoosting", "score > 0.5"),
    ("12.5: Voting Classifier", "voting = VotingClassifier(estimators=[...])", "score > 0.5"),
    ("12.6: Stacking", "stacking = StackingClassifier(estimators=[...])", "score > 0.5"),
    ("12.7: Ensemble Comparison", "Compare RF, GB, and Voting", "len(scores) == 3")
]

for i, (title, desc, check) in enumerate(koans, 1):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": f"## KOAN {title}\n{desc}"})
    
    if i == 1:
        code = "def koan_1():\n    X, y = make_classification(n_samples=100, n_features=10, random_state=42)\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n    rf = RandomForestClassifier(n_estimators=100, random_state=42)\n    rf.fit(X_train, y_train)\n    return rf.score(X_test, y_test)\n\n@validator.koan(1, 'Random Forest', difficulty='Advanced')\ndef validate():\n    score = koan_1()\n    assert score > 0.5\n    print(f'✓ RF score: {score:.3f}')\nvalidate()"
    elif i == 2:
        code = "def koan_2():\n    X, y = make_classification(n_samples=100, n_features=10, random_state=42)\n    rf = RandomForestClassifier(n_estimators=100, random_state=42)\n    rf.fit(X, y)\n    return rf.feature_importances_\n\n@validator.koan(2, 'Feature Importance', difficulty='Advanced')\ndef validate():\n    imp = koan_2()\n    assert len(imp) == 10\n    print(f'✓ Top importance: {imp.max():.3f}')\nvalidate()"
    elif i == 3:
        code = "def koan_3():\n    X, y = make_classification(n_samples=100, n_features=10, random_state=42)\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)\n    gb.fit(X_train, y_train)\n    return gb.score(X_test, y_test)\n\n@validator.koan(3, 'Gradient Boosting', difficulty='Advanced')\ndef validate():\n    score = koan_3()\n    assert score > 0.5\n    print(f'✓ GB score: {score:.3f}')\nvalidate()"
    elif i == 4:
        code = "def koan_4():\n    X, y = make_classification(n_samples=100, n_features=10, random_state=42)\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n    try:\n        import xgboost as xgb\n        model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')\n    except ImportError:\n        model = GradientBoostingClassifier(n_estimators=100, random_state=42)\n    model.fit(X_train, y_train)\n    return model.score(X_test, y_test)\n\n@validator.koan(4, 'XGBoost', difficulty='Advanced')\ndef validate():\n    score = koan_4()\n    assert score > 0.5\n    print(f'✓ XGB score: {score:.3f}')\nvalidate()"
    elif i == 5:
        code = "def koan_5():\n    X, y = make_classification(n_samples=100, n_features=10, random_state=42)\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n    rf = RandomForestClassifier(n_estimators=50, random_state=42)\n    gb = GradientBoostingClassifier(n_estimators=50, random_state=42)\n    lr = LogisticRegression(random_state=42, max_iter=1000)\n    voting = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('lr', lr)], voting='soft')\n    voting.fit(X_train, y_train)\n    return voting.score(X_test, y_test)\n\n@validator.koan(5, 'Voting Classifier', difficulty='Advanced')\ndef validate():\n    score = koan_5()\n    assert score > 0.5\n    print(f'✓ Voting score: {score:.3f}')\nvalidate()"
    elif i == 6:
        code = "def koan_6():\n    X, y = make_classification(n_samples=100, n_features=10, random_state=42)\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n    estimators = [('rf', RandomForestClassifier(n_estimators=50, random_state=42)), ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42