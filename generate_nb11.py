#!/usr/bin/env python3
"""Generate Notebook 11: Dimensionality Reduction with real koans."""

import json

def create_notebook_11():
    """Create notebook 11 with all 8 real koans."""
    
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
    
    # Add all cells
    cells = get_all_cells()
    nb["cells"] = cells
    
    # Write to file
    with open('koans/notebooks/11_dimensionality_reduction.ipynb', 'w') as f:
        json.dump(nb, f, indent=2)
    
    print("✓ Created notebook 11 with 8 real koans")

def get_all_cells():
    """Return all cells for the notebook."""
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Notebook 11: Dimensionality Reduction\n",
            "\n",
            "**Prerequisites**: Clustering  \n",
            "**Difficulty**: Intermediate-Advanced  \n",
            "**Time**: 2-3 hours\n",
            "\n",
            "Master dimensionality reduction techniques including PCA, t-SNE, and understand the curse of dimensionality."
        ]
    })
    
    # Setup cell
    cells.append({
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
    
    # Koan 11.1: PCA Basics
    cells.extend(get_koan_11_1())
    
    # Koan 11.2: Explained Variance
    cells.extend(get_koan_11_2())
    
    # Koan 11.3: Scree Plot
    cells.extend(get_koan_11_3())
    
    # Koan 11.4: PCA Visualization
    cells.extend(get_koan_11_4())
    
    # Koan 11.5: Feature Loadings
    cells.extend(get_koan_11_5())
    
    # Koan 11.6: t-SNE
    cells.extend(get_koan_11_6())
    
    # Koan 11.7: UMAP (simplified - no external dependency)
    cells.extend(get_koan_11_7())
    
    # Koan 11.8: Curse of Dimensionality
    cells.extend(get_koan_11_8())
    
    # Progress cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Congratulations!\n",
            "\n",
            "You've completed Dimensionality Reduction! You've mastered:\n",
            "- Principal Component Analysis (PCA)\n",
            "- Explained variance analysis\n",
            "- Component selection techniques\n",
            "- Visualization with dimensionality reduction\n",
            "- Understanding the curse of dimensionality"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "progress = tracker.get_notebook_progress('11_dimensionality_reduction')\n",
            "print(f'\\nNotebook Progress: {progress:.1f}%')\n",
            "print('\\nNext: Notebook 12 - Ensemble Methods')"
        ]
    })
    
    return cells

def get_koan_11_1():
    """Koan 11.1: Principal Component Analysis."""
    return [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## KOAN 11.1: Principal Component Analysis\n",
                "\n",
                "**Concept**: Linear dimensionality reduction using PCA  \n",
                "**Key Skills**: `PCA`, `fit_transform()`\n",
                "\n",
                "PCA finds orthogonal directions of maximum variance in the data.\n",
                "\n",
                "**Your Task**: Apply PCA to reduce the Iris dataset from 4 features to 2 components."
            ]
        },
        {
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
                "    pca = PCA(n_components=____)\n",
                "    \n",
                "    # TODO: Fit and transform the data\n",
                "    X_reduced = pca.____(X)\n",
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
        }
    ]

def get_koan_11_2():
    """Koan 11.2: Explained Variance."""
    return [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## KOAN 11.2: Explained Variance\n",
                "\n",
                "**Concept**: Understanding information retention  \n",
                "**Key Skills**: `explained_variance_ratio_`, cumulative variance\n",
                "\n",
                "Each principal component explains a percentage of the total variance.\n",
                "\n",
                "**Your Task**: Calculate the cumulative explained variance ratio for digits data."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def koan_11_2():\n",
                "    \"\"\"Calculate cumulative explained variance.\"\"\"\n",
                "    digits = load_digits()\n",
                "    X = digits.