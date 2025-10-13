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

# Title
nb["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": "# Notebook 11: Dimensionality Reduction\n\nMaster PCA, t-SNE and dimensionality concepts."
})

# Setup
nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": "import sys\nsys.path.append('../..')\nimport numpy as np\nfrom sklearn.decomposition import PCA\nfrom sklearn.manifold import TSNE\nfrom sklearn.datasets import load_digits, load_iris\nfrom koans.core.validator import KoanValidator\nfrom koans.core.progress import ProgressTracker\nvalidator = KoanValidator('11_dimensionality_reduction')\ntracker = ProgressTracker()\nprint('Setup complete!')"
})

# Koan 1
nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": "## KOAN 11.1: PCA Basics\nApply PCA to reduce Iris to 2D."})
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "def koan_1():\n    iris = load_iris()\n    X = iris.data\n    pca = PCA(n_components=2)\n    X_2d = pca.fit_transform(X)\n    return X_2d\n\n@validator.koan(1, 'PCA Basics', difficulty='Intermediate-Advanced')\ndef validate():\n    result = koan_1()\n    assert result.shape == (150, 2)\n    print('✓ Reduced to 2D')\nvalidate()"})

# Koan 2
nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": "## KOAN 11.2: Explained Variance\nGet cumulative explained variance."})
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "def koan_2():\n    digits = load_digits()\n    pca = PCA(n_components=10).fit(digits.data)\n    return np.cumsum(pca.explained_variance_ratio_)\n\n@validator.koan(2, 'Explained Variance', difficulty='Intermediate-Advanced')\ndef validate():\n    result = koan_2()\n    assert len(result) == 10\n    print(f'✓ 10 components: {result[-1]*100:.1f}% variance')\nvalidate()"})

# Koan 3
nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": "## KOAN 11.3: Component Selection\nFind components for 95% variance."})
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "def koan_3():\n    digits = load_digits()\n    pca = PCA(n_components=50).fit(digits.data)\n    cum = np.cumsum(pca.explained_variance_ratio_)\n    return np.argmax(cum >= 0.95) + 1\n\n@validator.koan(3, 'Component Selection', difficulty='Intermediate-Advanced')\ndef validate():\n    result = koan_3()\n    assert 10 <= result <= 40\n    print(f'✓ Need {result} components')\nvalidate()"})

# Koan 4
nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": "## KOAN 11.4: PCA Visualization\nReduce digits to 2D."})
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "def koan_4():\n    digits = load_digits()\n    pca = PCA(n_components=2)\n    return pca.fit_transform(digits.data), digits.target\n\n@validator.koan(4, 'PCA Visualization', difficulty='Intermediate-Advanced')\ndef validate():\n    X_2d, y = koan_4()\n    assert X_2d.shape == (1797, 2)\n    print('✓ Reduced digits to 2D')\nvalidate()"})

# Koan 5
nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": "## KOAN 11.5: Feature Loadings\nGet component loadings."})
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "def koan_5():\n    iris = load_iris()\n    pca = PCA(n_components=2).fit(iris.data)\n    return pca.components_\n\n@validator.koan(5, 'Feature Loadings', difficulty='Intermediate-Advanced')\ndef validate():\n    result = koan_5()\n    assert result.shape == (2, 4)\n    print('✓ Got loadings')\nvalidate()"})

# Koan 6
nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": "## KOAN 11.6: t-SNE\nApply t-SNE for visualization."})
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "def koan_6():\n    digits = load_digits()\n    X = digits.data[:500]\n    tsne = TSNE(n_components=2, random_state=42)\n    return tsne.fit_transform(X)\n\n@validator.koan(6, 't-SNE', difficulty='Intermediate-Advanced')\ndef validate():\n    result = koan_6()\n    assert result.shape[1] == 2\n    print('✓ Applied t-SNE')\nvalidate()"})

# Koan 7
nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": "## KOAN 11.7: Standardization\nScale before PCA."})
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "def koan_7():\n    from sklearn.preprocessing import StandardScaler\n    iris = load_iris()\n    scaler = StandardScaler()\n    return scaler.fit_transform(iris.data)\n\n@validator.koan(7, 'Standardization', difficulty='Intermediate-Advanced')\ndef validate():\n    result = koan_7()\n    assert abs(result.mean()) < 0.1\n    print('✓ Standardized data')\nvalidate()"})

# Koan 8
nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": "## KOAN 11.8: Curse of Dimensionality\nCompare distance variance."})
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "def koan_8():\n    np.random.seed(42)\n    pts_high = np.random.rand(100, 100)\n    pts_low = np.random.rand(100, 2)\n    from scipy.spatial.distance import pdist\n    return np.std(pdist(pts_high)), np.std(pdist(pts_low))\n\n@validator.koan(8, 'Curse of Dimensionality', difficulty='Intermediate-Advanced')\ndef validate():\n    std_high, std_low = koan_8()\n    assert std_high < std_low\n    print(f'✓ High-D std: {std_high:.3f}, Low-D std: {std_low:.3f}')\nvalidate()"})

# Progress
nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": "## Congratulations!\nCompleted Dimensionality Reduction!"})
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "progress = tracker.get_notebook_progress('11_dimensionality_reduction')\nprint(f'Progress: {progress}%')"})

with open('koans/notebooks/11_dimensionality_reduction.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("✓ Created notebook 11 with 8 real koans")