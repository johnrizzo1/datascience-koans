#!/usr/bin/env python3
"""Recreate notebooks 11-15 with real koans from KOAN_CATALOG."""

import json
import os

def create_nb11():
    """Notebook 11: Dimensionality Reduction (8 koans)."""
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.8.0"}}, "nbformat": 4, "nbformat_minor": 4}
    
    # Title
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": ["# Notebook 11: Dimensionality Reduction\n\n**Prerequisites**: Clustering\n**Difficulty**: Intermediate-Advanced\n**Time**: 2-3 hours\n\nMaster PCA, t-SNE, and the curse of dimensionality."]})
    
    # Setup
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import sys\nsys.path.append('../..')\nimport numpy as np\nimport pandas as pd\nfrom sklearn.decomposition import PCA\nfrom sklearn.manifold import TSNE\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.datasets import load_digits, load_iris\nfrom koans.core.validator import KoanValidator\nfrom koans.core.progress import ProgressTracker\n\nvalidator = KoanValidator('11_dimensionality_reduction')\ntracker = ProgressTracker()\nprint('Setup complete!')"]})
    
    # 8 koans
    koans = [
        ("11.1: PCA Basics", "Apply PCA to reduce Iris to 2D", "pca = PCA(n_components=2)\nX_reduced = pca.fit_transform(X)", "result.shape == (150, 2)"),
        ("11.2: Explained Variance", "Get cumulative explained variance", "cumulative = np.cumsum(pca.explained_variance_ratio_)", "len(result) == 10"),
        ("11.3: Scree Plot", "Find components for 95% variance", "n_comp = np.argmax(cumulative >= 0.95) + 1", "10 <= result <= 40"),
        ("11.4: PCA Visualization", "Reduce digits to 2D", "pca = PCA(n_components=2)\nX_2d = pca.fit_transform(X)", "result[0].shape[1] == 2"),
        ("11.5: Feature Loadings", "Get PCA component loadings", "loadings = pca.components_", "result.shape[0] == 2"),
        ("11.6: t-SNE", "Apply t-SNE for visualization", "tsne = TSNE(n_components=2, random_state=42)\nX_tsne = tsne.fit_transform(X)", "result.shape[1] == 2"),
        ("11.7: Standardization First", "Scale before PCA", "scaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)", "abs(result.mean()) < 0.1"),
        ("11.8: Curse of Dimensionality", "Compare distances in high vs low dim", "dist_high = np.std(distances_high)\ndist_low = np.std(distances_low)", "result > 0")
    ]
    
    for i, (title, desc, hint, check) in enumerate(koans, 1):
        nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [f"## KOAN {title}\n**Objective**: {desc}\n**Difficulty**: Intermediate-Advanced"]})
        nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [f"def koan_{i}():\n    # TODO: {desc}\n    # Hint: {hint}\n    pass\n\n@validator.koan({i}, '{title}', difficulty='Intermediate-Advanced')\ndef validate():\n    result = koan_{i}()\n    assert {check}\n\nvalidate()"]})
    
    # Progress
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": ["## Congratulations!\nYou've completed Dimensionality Reduction!"]})
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["progress = tracker.get_notebook_progress('11_dimensionality_reduction')\nprint(f'Progress: {progress}%')"]})
    
    return nb

def create_nb12():
    """Notebook 12: Ensemble Methods (7 koans)."""
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.8.0"}}, "nbformat": 4, "nbformat_minor": 4}
    
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": ["# Notebook 12: Ensemble Methods\n\n**Prerequisites**: Dimensionality Reduction\n**Difficulty**: Advanced\n**Time**: 2-3 hours\n\nMaster Random Forests, Gradient Boosting, and ensemble techniques."]})
    
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import sys\nsys.path.append('../..')\nimport numpy as np\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.datasets import make_classification\nfrom koans.core.validator import KoanValidator\nfrom koans.core.progress import ProgressTracker\n\nvalidator = KoanValidator('12_ensemble_methods')\ntracker = ProgressTracker()\nprint('Setup complete!')"]})
    
    koans = [
        ("12.1: Random Forest", "Create Random Forest classifier", "rf = RandomForestClassifier(n_estimators=100)", "hasattr(result, 'n_estimators')"),
        ("12.2: Feature Importance", "Get feature importances", "importances = rf.feature_importances_", "len(result) > 0"),
        ("12.3: Gradient Boosting", "Train Gradient Boosting", "gb = GradientBoostingClassifier(n_estimators=100)", "result > 0.5"),
        ("12.4: XGBoost", "Use XGBoost (if available)", "import xgboost as xgb\nmodel = xgb.XGBClassifier()", "result > 0"),
        ("12.5: Voting Classifier", "Combine models with voting", "voting = VotingClassifier(estimators=[...])", "hasattr(result, 'estimators')"),
        ("12.6: Stacking", "Stack multiple models", "from sklearn.ensemble import StackingClassifier", "result > 0.5"),
        ("12.7: Ensemble Comparison", "Compare ensemble methods", "scores = [rf_score, gb_score, voting_score]", "len(result) == 3")
    ]
    
    for i, (title, desc, hint, check) in enumerate(koans, 1):
        nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [f"## K