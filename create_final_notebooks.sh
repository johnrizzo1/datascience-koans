#!/bin/bash
# Create the final 7 notebooks (09-15)

echo "Creating notebooks 09-15..."

# Each notebook follows the same pattern but with different content
# NB09: Model Evaluation (10 koans)
# NB10: Clustering (8 koans)
# NB11: Dimensionality Reduction (8 koans)  
# NB12: Ensemble Methods (7 koans)
# NB13: Hyperparameter Tuning (7 koans)
# NB14: Pipelines (5 koans)
# NB15: Ethics and Bias (5 koans)

python3 << 'PYEOF'
import json

# Template function
def create_nb(num, title, nb_id, koans):
    cells = []
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [
        f"# {title} - Data Science Koans\n", "\n", f"Master {title.lower()}!\n", "\n",
        "## How to Use\n", "1. Read koans\n", "2. Complete TODOs\n",
        "3. Run validation\n", "4. Iterate"]})
    
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], 
        "source": ["# Setup\n", "import sys\n", "sys.path.append('../..')\n",
        "import numpy as np\n", "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from koans.core.validator import KoanValidator\n",
        "from koans.core.progress import ProgressTracker\n", "\n",
        f"validator = KoanValidator('{nb_id}')\n", "tracker = ProgressTracker()\n",
        "print('Setup complete!')\n",
        f"print(f\"Progress: {{tracker.get_notebook_progress('{nb_id}')}}%\")"]})
    
    for i, (t, o, d, c, v) in enumerate(koans, 1):
        cells.append({"cell_type": "markdown", "metadata": {}, 
            "source": [f"## KOAN {num}.{i}: {t}\n**Objective**: {o}\n**Difficulty**: {d}"]})
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [f"{c}\n\n@validator.koan({i}, \"{t}\", difficulty=\"{d}\")\ndef validate():\n    {v}\nvalidate()"]})
    
    cells.append({"cell_type": "markdown", "metadata": {}, 
        "source": [f"## Congratulations!\n\nYou completed {title}!"]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [f"progress = tracker.get_notebook_progress('{nb_id}')\n",
        "print(f'Final Progress: {progress}%')\n", "if progress == 100:\n",
        f"    print('Mastered {title}!')"]})
    
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py", "mimetype": "text/x-python", "name": "python",
        "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.8.0"}},
        "nbformat": 4, "nbformat_minor": 4}
    
    with open(f'koans/notebooks/{nb_id}.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Created {nb_id}")

# NB09 koans
nb09 = [
    ("Cross Validation", "K-fold", "Intermediate", "def do_cv():\n    from sklearn.model_selection import cross_val_score\n    from sklearn.linear_model import LogisticRegression\n    X = [[1],[2],[3],[4],[5]]\n    y = [0,0,1,1,1]\n    # TODO: 3-fold CV\n    pass", "scores = do_cv()\nassert len(scores) == 3"),
    ("CV Scores", "Mean score", "Intermediate", "def cv_mean():\n    scores = [0.8, 0.9, 0.85]\n    # TODO: Return mean\n    pass", "result = cv_mean()\nassert result == 0.85"),
    ("Train Val Test", "3-way split", "Intermediate", "def split_3way():\n    X = np.arange(100).reshape(-1, 1)\n    # TODO: 60/20/20 split\n    pass", "tr, v, te = split_3way()\nassert len(tr) == 60"),
    ("Stratified Split", "Preserve ratios", "Intermediate", "def stratified():\n    from sklearn.model_selection import StratifiedKFold\n    # TODO: Create 3-fold\n    pass", "skf = stratified()\nassert skf.n_splits == 3"),
    ("Learning Curve", "Training vs CV", "Intermediate", "def learning_curve():\n    # TODO: Return True\n    return True", "result = learning_curve()\nassert result == True"),
    ("ROC Curve", "TPR vs FPR", "Intermediate", "def roc():\n    from sklearn.metrics import roc_curve\n    y = [0,0,1,1]\n    pred = [0.1,0.4,0.6,0.9]\n    # TODO: Compute ROC\n    pass", "fpr, tpr, _ = roc()\nassert len(fpr) > 0"),
    ("AUC Score", "Area under curve", "Intermediate", "def auc():\n    from sklearn.metrics import roc_auc_score\n    y = [0,1,1,0]\n    pred = [0.1,0.9,0.8,0.2]\n    # TODO: Compute AUC\n    pass", "result = auc()\nassert result > 0.5"),
    ("Validation Curve", "Param vs score", "Intermediate", "def val_curve():\n    # TODO: Return True\n    return True", "result = val_curve()\nassert result == True"),
    ("Overfitting Check", "Train vs test gap", "Intermediate", "def overfit_check():\n    train_score = 0.99\n    test_score = 0.65\n    # TODO: Return True if gap > 0.2\n    pass", "result = overfit_check()\nassert result == True"),
    ("Bootstrap Sample", "Resample with replacement", "Intermediate", "def bootstrap():\n    data = [1,2,3,4,5]\n    # TODO: Sample with replacement\n    pass", "result = bootstrap()\nassert len(result) == 5")
]

# NB10 koans
nb10 = [
    ("KMeans Clustering", "K centroids", "Advanced", "def kmeans_cluster():\n    from sklearn.cluster import KMeans\n    X = [[1,2],[2,3],[8,9],[9,10]]\n    # TODO: Fit K=2\n    pass", "model = kmeans_cluster()\nassert hasattr(model, 'labels_')"),
    ("Cluster Labels", "Assign points", "Advanced", "def get_labels():\n    