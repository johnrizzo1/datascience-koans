#!/usr/bin/env python3
import json

cells = []
cells.append({"cell_type": "markdown", "metadata": {}, "source": [
    "# Clustering - Data Science Koans\n\n",
    "Master clustering algorithms!\n\n",
    "## How to Use\n1. Read koans\n2. Complete TODOs\n3. Validate\n4. Iterate"]})

cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "# Setup\n", "import sys\n", "sys.path.append('../..')\n",
    "import numpy as np\n", "import pandas as pd\n",
    "from sklearn.cluster import KMeans, AgglomerativeClustering\n",
    "from sklearn.metrics import silhouette_score\n",
    "from koans.core.validator import KoanValidator\n",
    "from koans.core.progress import ProgressTracker\n\n",
    "validator = KoanValidator('10_clustering')\n",
    "tracker = ProgressTracker()\n", "print('Setup complete!')\n",
    "print(f\"Progress: {tracker.get_notebook_progress('10_clustering')}%\")"]})

koans = [
    ("10.1: KMeans Fit", "Cluster data", "Advanced",
     "def kmeans_fit():\n    X = np.array([[1,2],[2,3],[8,9],[9,10]])\n    model = KMeans(n_clusters=2, random_state=42, n_init=10)\n    # TODO: Fit model\n    pass",
     "model = kmeans_fit()\nassert hasattr(model, 'labels_')"),
    
    ("10.2: Get Labels", "Cluster assignments", "Advanced",
     "def get_labels():\n    X = np.array([[1,2],[2,3],[8,9],[9,10]])\n    model = KMeans(n_clusters=2, random_state=42, n_init=10)\n    model.fit(X)\n    # TODO: Return labels_\n    pass",
     "labels = get_labels()\nassert len(labels) == 4"),
    
    ("10.3: Centroids", "Cluster centers", "Advanced",
     "def get_centers():\n    X = np.array([[1,2],[2,3],[8,9],[9,10]])\n    model = KMeans(n_clusters=2, random_state=42, n_init=10)\n    model.fit(X)\n    # TODO: Return cluster_centers_\n    pass",
     "centers = get_centers()\nassert centers.shape == (2, 2)"),
    
    ("10.4: Predict", "Assign new point", "Advanced",
     "def predict_new():\n    X = np.array([[1,2],[2,3],[8,9],[9,10]])\n    model = KMeans(n_clusters=2, random_state=42, n_init=10)\n    model.fit(X)\n    # TODO: Predict [[1.5, 2.5]]\n    pass",
     "cluster = predict_new()\nassert cluster[0] in [0, 1]"),
    
    ("10.5: Inertia", "Sum of squares", "Advanced",
     "def get_inertia():\n    X = np.array([[1,2],[2,3],[8,9],[9,10]])\n    model = KMeans(n_clusters=2, random_state=42, n_init=10)\n    model.fit(X)\n    # TODO: Return inertia_\n    pass",
     "inertia = get_inertia()\nassert inertia > 0"),
    
    ("10.6: Silhouette", "Cluster quality", "Advanced",
     "def calc_silhouette():\n    X = np.array([[1,2],[2,3],[8,9],[9,10]])\n    model = KMeans(n_clusters=2, random_state=42, n_init=10)\n    labels = model.fit_predict(X)\n    # TODO: Calculate silhouette_score\n    pass",
     "score = calc_silhouette()\nassert -1 <= score <= 1"),
    
    ("10.7: Hierarchical", "Agglomerative", "Advanced",
     "def hierarchical():\n    X = np.array([[1,2],[2,3],[8,9],[9,10]])\n    model = AgglomerativeClustering(n_clusters=2)\n    # TODO: Fit model\n    pass",
     "model = hierarchical()\nassert hasattr(model, 'labels_')"),
    
    ("10.8: Elbow Method", "Choose K", "Advanced",
     "def elbow_concept():\n    # Elbow: plot inertia vs K, choose where curve bends\n    # TODO: Return True\n    pass",
     "result = elbow_concept()\nassert result == True")
]

for i, (title, obj, diff, code, val) in enumerate(koans, 1):
    cells.append({"cell_type": "markdown", "metadata": {}, 
        "source": [f"## KOAN {title}\n**Objective**: {obj}\n**Difficulty**: {diff}"]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [f"{code}\n\n@validator.koan({i}, \"{title.split(': ')[1]}\", difficulty=\"{diff}\")\ndef validate():\n    {val}\nvalidate()"]})

cells.append({"cell_type": "markdown", "metadata": {}, 
    "source": ["## Congratulations!\n\nYou completed Clustering!"]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": ["progress = tracker.get_notebook_progress('10_clustering')\n",
    "print(f'Final Progress: {progress}%')\n", "if progress == 100:\n",
    "    print('Excellent! Mastered Clustering!')"]})

notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"codemirror_mode": {"name": "ipython", "version": 3},
    "file_extension": ".py", "mimetype": "text/x-python", "name": "python",
    "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.8.0"}},
    "nbformat": 4, "nbformat_minor": 4}

with open('koans/notebooks/10_clustering.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
print("Notebook 10 created!")