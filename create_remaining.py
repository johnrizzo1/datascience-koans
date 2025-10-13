#!/usr/bin/env python3
import json

def mk_nb(num, title, koans_data):
    """Create notebook with given koans."""
    nb = {
        "cells": [],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.8.0"}},
        "nbformat": 4, "nbformat_minor": 4
    }
    
    # Title and setup
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": f"# Notebook {num}: {title}"})
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": koans_data["setup"]})
    
    # Add koans
    for i, (md, code) in enumerate(koans_data["koans"], 1):
        nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": f"## KOAN {num}.{i}: {md}"})
        nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code})
    
    # Progress
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": f"## Congratulations!\nCompleted {title}!"})
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": f"progress = tracker.get_notebook_progress('{num}_{title.lower().replace(' ', '_')}')\nprint(f'Progress: {{progress}}%')"})
    
    return nb

# Notebook 12: Ensemble Methods
nb12_data = {
    "setup": "import sys\nsys.path.append('../..')\nimport numpy as np\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.datasets import make_classification\nfrom koans.core.validator import KoanValidator\nfrom koans.core.progress import ProgressTracker\nvalidator = KoanValidator('12_ensemble_methods')\ntracker = ProgressTracker()\nprint('Setup complete!')",
    "koans": [
        ("Random Forest", "def koan_1():\n    X, y = make_classification(n_samples=100, n_features=10, random_state=42)\n    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n    rf = RandomForestClassifier(n_estimators=100, random_state=42)\n    rf.fit(X_train, y_train)\n    return rf.score(X_test, y_test)\n\n@validator.koan(1, 'Random Forest', difficulty='Advanced')\ndef validate():\n    score = koan_1()\n    assert score > 0.5\n    print(f'✓ RF score: {score:.3f}')\nvalidate()"),
        ("Feature Importance", "def koan_2():\n    X, y = make_classification(n_samples=100, n_features=10, random_state=42)\n    rf = RandomForestClassifier(n_estimators=100, random_state=42)\n    rf.fit(X, y)\n    return rf.feature_importances_\n\n@validator.koan(2, 'Feature Importance', difficulty='Advanced')\ndef validate():\n    imp = koan_2()\n    assert len(imp) == 10\n    print('✓ Got importances')\nvalidate()"),
        ("Gradient Boosting", "def koan_3():\n    X, y = make_classification(n_samples=100, n_features=10, random_state=42)\n    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)\n    gb.fit(X_train, y_train)\n    return gb.score(X_test, y_test)\n\n@validator.koan(3, 'Gradient Boosting', difficulty='Advanced')\ndef validate():\n    score = koan_3()\n    assert score > 0.5\n    print(f'✓ GB score: {score:.3f}')\nvalidate()"),
        ("XGBoost", "def koan_4():\n    X, y = make_classification(n_samples=100, n_features=10, random_state=42)\n    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n    try:\n        import xgboost as xgb\n        model = xgb.XGBClassifier(n_estimators=100, random_state=42)\n    except:\n        model = GradientBoostingClassifier(n_estimators=100, random_state=42)\n    model.fit(X_train, y_train)\n    return model.score(X_test, y_test)\n\n@validator.koan(4, 'XGBoost', difficulty='Advanced')\ndef validate():\n    score = koan_4()\n    assert score > 0.5\n    print(f'✓ Score: {score:.3f}')\nvalidate()"),
        ("Voting Classifier", "def koan_5():\n    X, y = make_classification(n_samples=100, n_features=10, random_state=42)\n    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n    rf = RandomForestClassifier(n_estimators=50, random_state=42)\n    gb = GradientBoostingClassifier(n_estimators=50, random_state=42)\n    lr = LogisticRegression(random_state=42, max_iter=1000)\n    voting = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('lr', lr)], voting='soft')\n    voting.fit(X_train, y_train)\n    return voting.score(X_test, y_test)\n\n@validator.koan(5, 'Voting', difficulty='Advanced')\ndef validate():\n    score = koan_5()\n    assert score > 0.5\n    print(f'✓ Voting score: {score:.3f}')\nvalidate()"),
        ("Stacking", "def koan_6():\n    X, y = make_classification(n_samples=100, n_features=10, random_state=42)\n    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n    estimators = [('rf', RandomForestClassifier(n_estimators=50, random_state=42)), ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))]\n    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())\n    stacking.fit(X_train, y_train)\n    return stacking.score(X_test, y_test)\n\n@validator.koan(6, 'Stacking', difficulty='Advanced')\ndef validate():\n    score = koan_6()\n    assert score > 0.5\n    print
