#!/usr/bin/env python3
"""Generate Notebook 11: Dimensionality Reduction with real koans."""

import json

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Title cell
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Notebook 11: Dimensionality Reduction\n\n",
            "**Prerequisites**: Clustering\n",
            "**Difficulty**: Intermediate-Advanced\n",
            "**Time**: 2-3 hours\n\n",
            "In this notebook, you'll master dimensionality reduction techniques including PCA, t-SNE, and understand the curse of dimensionality."
        ]
    })
    
    # Setup cell
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "sys.path.append('../..')\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "from sklearn.decomposition import PCA\n",
            "from sklearn.manifold import TSNE\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.datasets import load_digits, load_iris\n",
            "import matplotlib.pyplot as plt\n",
            "from koans.core.validator import KoanValidator\n",
            "from koans.core.progress import ProgressTracker\n",
            "\n",
            "validator = KoanValidator('11_dimensionality_reduction')\n",
            "tracker = ProgressTracker()\n",
            "print('Setup complete!')"
        ]
    })
    
    # Koan 11.1: Principal Component Analysis
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## KOAN 11.1: Principal Component Analysis\n\n",
            "**Concept**: Linear dimensionality reduction using PCA\n",
            "**Key Skills**: `PCA`, `fit_transform()`, understanding components\n\n",
            "PCA finds orthogonal directions of maximum variance in the data.\n\n",
            "**Your Task**: Apply PCA to reduce the Iris dataset from 4 features to 2 components."
        ]
    })
    
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def koan_11_1():\n",
            "    \"\"\"Apply PCA to reduce Iris dataset to 2 components.\"\"\"\n",
            "    iris = load_iris()\n",
            "    X = iris.data\n",
            "    \n",
            "    # TODO: Create a PCA object with 2 components\n",
            "    pca = ____\n",
            "    \n",
            "    # TODO: Fit and transform the data\n",
            "    X_reduced = ____\n",
            "    \n",
            "    return X_reduced\n",
            "\n",
            "@validator.koan(1, 'Principal Component Analysis', difficulty='Intermediate-Advanced')\n",
            "def validate():\n",
            "    result = koan_11_1()\n",
            "    assert result.shape == (150, 2), \"Should reduce to 2 components\"\n",
            "    assert isinstance(result, np.ndarray), \"Should return numpy array\"\n",
            "    print(\"✓ Successfully applied PCA to reduce dimensions from 4 to 2\")\n",
            "\n",
            "validate()"
        ]
    })
    
    # Koan 11.2: Explained Variance
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## KOAN 11.2: Explained Variance\n\n",
            "**Concept**: Understanding how much information each component retains\n",
            "**Key Skills**: `explained_variance_ratio_`, cumulative variance\n\n",
            "Each principal component explains a certain percentage of the total variance.\n\n",
            "**Your Task**: Calculate the cumulative explained variance ratio for PCA on digits data."
        ]
    })
    
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def koan_11_2():\n",
            "    \"\"\"Calculate cumulative explained variance for digits dataset.\"\"\"\n",
            "    digits = load_digits()\n",
            "    X = digits.data  # 64 features (8x8 pixels)\n",
            "    \n",
            "    # TODO: Create PCA with 10 components\n",
            "    pca = ____\n",
            "    pca.fit(X)\n",
            "    \n",
            "    # TODO: Get the explained variance ratio\n",
            "    explained_var = ____\n",
            "    \n",
            "    # TODO: Calculate cumulative sum\n",
            "    cumulative_var = ____\n",
            "    \n",
            "    return cumulative_var\n",
            "\n",
            "@validator.koan(2, 'Explained Variance', difficulty='Intermediate-Advanced')\n",
            "def validate():\n",
            "    result = koan_11_2()\n",
            "    assert len(result) == 10, \"Should have 10 components\"\n",
            "    assert result[0] > 0 and result[-1] <= 1.0, \"Should be valid variance ratios\"\n",
            "    assert result[-1] > result[0], \"Should be cumulative\"\n",
            "    print(f\"✓ First 10 components explain {result[-1]*100:.1f}% of variance\")\n",
            "\n",
            "validate()"
        ]
    })
    
    # Koan 11.3: Scree Plot
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## KOAN 11.3: Scree Plot\n\n",
            "**Concept**: Visual method for selecting number of components\n",
            "**Key Skills**: Plotting variance, choosing components\n\n",
            "A scree plot shows variance explained by each component.\n\n",
            "**Your Task**: Determine how many components are needed to explain 95% of variance."
        ]
    })
    
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def koan_11_3():\n",
            "    \"\"\"Find number of components for 95% variance.\"\"\"\n",
            "    digits = load_digits()\n",
            "    X = digits.data\n",
            "    \n",
            "    # TODO: Create PCA with all components (or large number like 50)\n",
            "    pca = ____\n",
            "    pca.fit(X)\n",
            "    \n",
            "    # TODO: Calculate cumulative explained variance\n",
            