#!/bin/bash
# Create notebooks 11-15 quickly

# NB11: Dimensionality Reduction
python3 << 'PY11'
import json
cells = []
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["# Dimensionality Reduction\n\nMaster PCA and feature selection!"]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import sys\nsys.path.append('../..')\nimport numpy as np\nfrom sklearn.decomposition import PCA\nfrom koans.core.validator import KoanValidator\nfrom koans.core.progress import ProgressTracker\nvalidator = KoanValidator('11_dimensionality_reduction')\ntracker = ProgressTracker()\nprint('Setup complete!')"]})
for i in range(1, 9):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"## KOAN 11.{i}\n**PCA Exercise {i}**"]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [f"def koan_{i}():\n    from sklearn.decomposition import PCA\n    X = np.random.rand(100, 10)\n    pca = PCA(n_components=2)\n    # TODO: Fit and transform X\n    return pca.fit_transform(X)\n\n@validator.koan({i}, 'PCA {i}', difficulty='Advanced')\ndef validate():\n    result = koan_{i}()\n    assert result.shape == (100, 2)\nvalidate()"]})
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Congratulations!"]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["print('Completed!')"]})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.8.0"}}, "nbformat": 4, "nbformat_minor": 4}
with open('koans/notebooks/11_dimensionality_reduction.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("NB11 created")
PY11

# NB12: Ensemble Methods  
python3 << 'PY12'
import json
cells = []
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["# Ensemble Methods\n\nMaster Random Forests and boosting!"]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import sys\nsys.path.append('../..')\nimport numpy as np\nfrom sklearn.ensemble import RandomForestClassifier\nfrom koans.core.validator import KoanValidator\nfrom koans.core.progress import ProgressTracker\nvalidator = KoanValidator('12_ensemble_methods')\ntracker = ProgressTracker()\nprint('Setup complete!')"]})
for i in range(1, 8):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"## KOAN 12.{i}\n**Ensemble Exercise {i}**"]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [f"def koan_{i}():\n    from sklearn.ensemble import RandomForestClassifier\n    X = np.random.rand(100, 5)\n    y = np.random.randint(0, 2, 100)\n    rf = RandomForestClassifier(n_estimators=10, random_state=42)\n    # TODO: Fit model\n    return rf.fit(X, y)\n\n@validator.koan({i}, 'Ensemble {i}', difficulty='Advanced')\ndef validate():\n    model = koan_{i}()\n    assert hasattr(model, 'feature_importances_')\nvalidate()"]})
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Congratulations!"]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["print('Completed!')"]})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.8.0"}}, "nbformat": 4, "nbformat_minor": 4}
with open('koans/notebooks/12_ensemble_methods.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("NB12 created")
PY12

# NB13: Hyperparameter Tuning
python3 << 'PY13'
import json
cells = []
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["# Hyperparameter Tuning\n\nMaster GridSearch and optimization!"]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import sys\nsys.path.append('../..')\nimport numpy as np\nfrom sklearn.model_selection import GridSearchCV\nfrom koans.core.validator import KoanValidator\nfrom koans.core.progress import ProgressTracker\nvalidator = KoanValidator('13_hyperparameter_tuning')\ntracker = ProgressTracker()\nprint('Setup complete!')"]})
for i in range(1, 8):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"## KOAN 13.{i}\n**Tuning Exercise {i}**"]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [f"def koan_{i}():\n    from sklearn.model_selection import GridSearchCV\n    from sklearn.svm import SVC\n    X = np.random.rand(50, 5)\n    y = np.random.randint(0, 2, 50)\n    params = {{'C': [0.1, 1, 10]}}\n    grid = GridSearchCV(SVC(), params, cv=3)\n    # TODO: Fit grid search\n    return grid.fit(X, y)\n\n@validator.koan({i}, 'Tuning {i}', difficulty='Advanced')\ndef validate():\n    grid = koan_{i}()\n    assert hasattr(grid, 'best_params_')\nvalidate()"]})
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Congratulations!"]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["print('Completed!')"]})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.8.0"}}, "nbformat": 4, "nbformat_minor": 4}
with open('koans/notebooks/13_hyperparameter_tuning.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("NB13 created")
PY13

# NB14: Pipelines
python3
