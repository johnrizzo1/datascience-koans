#!/usr/bin/env python3
import json

cells = []
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["# Regression Basics - Data Science Koans\n", "\n", "Master regression!\n", "\n", "## How to Use\n", "1. Read each koan\n", "2. Complete TODOs\n", "3. Run validation\n", "4. Iterate"]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["# Setup\n", "import sys\n", "sys.path.append('../..')\n", "import numpy as np\n", "import pandas as pd\n", "from sklearn.linear_model import LinearRegression\n", "from sklearn.model_selection import train_test_split\n", "from koans.core.validator import KoanValidator\n", "from koans.core.progress import ProgressTracker\n", "\n", "validator = KoanValidator('07_regression_basics')\n", "tracker = ProgressTracker()\n", "print('Setup complete!')\n", "print(f\"Progress: {tracker.get_notebook_progress('07_regression_basics')}%\")"]})

koans = [
    ("7.1: Train Test Split", "Split data", "Intermediate", "def split_data():\n    X = np.array([[1], [2], [3], [4]])\n    y = np.array([2, 4, 6, 8])\n    # TODO: Split 75/25\n    pass", "X_train, X_test, y_train, y_test = split_data()\nassert len(X_train) == 3"),
    ("7.2: Fit Model", "Train regressor", "Intermediate", "def fit_model():\n    X = np.array([[1], [2], [3]])\n    y = np.array([2, 4, 6])\n    # TODO: Fit LinearRegression\n    pass", "model = fit_model()\nassert hasattr(model, 'coef_')"),
    ("7.3: Make Predictions", "Predict values", "Intermediate", "def predict():\n    model = LinearRegression()\n    model.fit([[1], [2]], [2, 4])\n    # TODO: Predict for [[3]]\n    pass", "result = predict()\nassert result[0] == 6.0"),
    ("7.4: Model Coefficients", "Get params", "Intermediate", "def get_coef():\n    model = LinearRegression()\n    model.fit([[1], [2], [3]], [2, 4, 6])\n    # TODO: Return coef_[0]\n    pass", "result = get_coef()\nassert result == 2.0"),
    ("7.5: MSE Calculation", "Mean squared error", "Intermediate", "def calc_mse():\n    y_true = np.array([1, 2, 3])\n    y_pred = np.array([1, 2, 3])\n    # TODO: Calculate MSE\n    pass", "result = calc_mse()\nassert result == 0.0"),
    ("7.6: R2 Score", "Coefficient determination", "Intermediate", "def calc_r2():\n    from sklearn.metrics import r2_score\n    y_true = [1, 2, 3]\n    y_pred = [1, 2, 3]\n    # TODO: Calculate R2\n    pass", "result = calc_r2()\nassert result == 1.0"),
    ("7.7: Residuals", "Calculate errors", "Intermediate", "def get_residuals():\n    y_true = np.array([1, 2, 3])\n    y_pred = np.array([1.1, 2.1, 2.9])\n    # TODO: Return y_true - y_pred\n    pass", "result = get_residuals()\nassert len(result) == 3"),
    ("7.8: Feature Scaling", "Standardize X", "Intermediate", "def scale_features():\n    from sklearn.preprocessing import StandardScaler\n    X = np.array([[1], [2], [3]])\n    # TODO: Fit and transform\n    pass", "result = scale_features()\nassert abs(result.mean()) < 0.01"),
    ("7.9: Multiple Features", "Multi-variate", "Intermediate", "def multi_feature():\n    X = np.array([[1, 2], [3, 4], [5, 6]])\n    y = np.array([3, 7, 11])\n    # TODO: Fit model\n    pass", "model = multi_feature()\nassert model.coef_.shape[0] == 2"),
    ("7.10: Polynomial Features", "Higher order", "Intermediate", "def poly_features():\n    from sklearn.preprocessing import PolynomialFeatures\n    X = np.array([[1], [2]])\n    # TODO: Transform degree=2\n    pass", "result = poly_features()\nassert result.shape[1] == 3")
]

for i, (title, obj, diff, code, val) in enumerate(koans, 1):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"## KOAN {title}\n**Objective**: {obj}\n**Difficulty**: {diff}"]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [f"{code}\n\n@validator.koan({i}, \"{title.split(': ')[1]}\", difficulty=\"{diff}\")\ndef validate():\n    {val}\nvalidate()"]})

cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Congratulations!\n\nYou completed Regression Basics!"]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["progress = tracker.get_notebook_progress('07_regression_basics')\n", "print(f'Final Progress: {progress}%')\n", "if progress == 100:\n", "    print('Excellent! You mastered Regression!')"]})

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.8.0"}}, "nbformat": 4, "nbformat_minor": 4}

with open('koans/notebooks/07_regression_basics.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
print("Notebook 07 created!")
