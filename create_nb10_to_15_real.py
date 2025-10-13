#!/usr/bin/env python3
"""Generate notebooks 10-15 with real content"""
import json

def create_nb(num, title, nb_id, koans_list):
    cells = []
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {title} - Data Science Koans\n\n"
            f"Master {title.lower()}!\n\n"
            "## How to Use\n"
            "1. Read each koan\n"
            "2. Complete TODOs\n"
            "3. Run validation\n"
            "4. Iterate"
        ]
    })
    
    imports = "import sys\nsys.path.append('../..')\nimport numpy as np\nimport pandas as pd\n"
    imports += "from koans.core.validator import KoanValidator\n"
    imports += "from koans.core.progress import ProgressTracker\n\n"
    imports += f"validator = KoanValidator('{nb_id}')\n"
    imports += "tracker = ProgressTracker()\n"
    imports += "print('Setup complete!')\n"
    imports += f"print(f\"Progress: {{tracker.get_notebook_progress('{nb_id}')}}%\")"
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": imports
    })
    
    for i, (t, o, d, c, v) in enumerate(koans_list, 1):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": f"## KOAN {num}.{i}: {t}\n**Objective**: {o}\n**Difficulty**: {d}"
        })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": f"{c}\n\n@validator.koan({i}, \"{t}\", difficulty=\"{d}\")\ndef validate():\n    {v}\nvalidate()"
        })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": f"## Congratulations!\n\nYou completed {title}!"
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": f"progress = tracker.get_notebook_progress('{nb_id}')\nprint(f'Final Progress: {{progress}}%')\nif progress == 100:\n    print('Excellent! Mastered {title}!')"
    })
    
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(f'koans/notebooks/{nb_id}.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Created {nb_id}")

# NB10: Clustering
nb10 = [
    ("KMeans Clustering", "Fit K-means", "Advanced",
     "def kmeans_cluster():\n    from sklearn.cluster import KMeans\n    X = np.array([[1,2],[2,3],[8,9],[9,10]])\n    # TODO: Fit KMeans with n_clusters=2\n    pass",
     "model = kmeans_cluster()\nassert hasattr(model, 'labels_')\nassert len(set(model.labels_)) == 2"),
    
    ("Cluster Labels", "Get assignments", "Advanced",
     "def get_labels():\n    from sklearn.cluster import KMeans\n    X = np.array([[1,2],[2,3],[8,9],[9,10]])\n    model = KMeans(n_clusters=2, random_state=42, n_init=10)\n    model.fit(X)\n    # TODO: Return labels_\n    pass",
     "labels = get_labels()\nassert len(labels) == 4"),
    
    ("Cluster Centers", "Get centroids", "Advanced",
     "def get_centers():\n    from sklearn.cluster import KMeans\n    X = np.array([[1,2],[2,3],[8,9],[9,10]])\n    model = KMeans(n_clusters=2, random_state=42, n_init=10)\n    model.fit(X)\n    # TODO: Return cluster_centers_\n    pass",
     "centers = get_centers()\nassert centers.shape == (2, 2)"),
    
    ("Predict Cluster", "Assign new point", "Advanced",
     "def predict_cluster():\n    from sklearn.cluster import KMeans\n    X = np.array([[1,2],[2,3],[8,9],[9,10]])\n    model = KMeans(n_clusters=2, random_state=42, n_init=10)\n    model.fit(X)\n    # TODO: Predict cluster for [[1.5, 2.5]]\n    pass",
     "cluster = predict_cluster()\nassert cluster[0] in [0, 1]"),
    
    ("Inertia", "Within-cluster sum", "Advanced",
     "def get_inertia():\n    from sklearn.cluster import KMeans\n    X = np.array([[1,2],[2,3],[8,9],[9,10]])\n    model = KMeans(n_clusters=2, random_state=42, n_init=10)\n    model.fit(X)\n    # TODO: Return inertia_\n    pass",
     "inertia = get_inertia()\nassert inertia > 0"),
    
    ("Elbow Method", "Choose K", "Advanced",
     "def elbow_concept():\n    # Elbow method: plot inertia vs K\n    # Choose K where inertia decrease slows\n    # TODO: Return True if understood\n    pass",
     "result = elbow_concept()\nassert result == True"),
    
    ("Silhouette Score", "Cluster quality", "Advanced",
     "def silhouette():\n    from sklearn.metrics import silhouette_score\n    from sklearn.cluster import KMeans\n    X = np.array([[1,2],[2,3],[8,9],[9,10]])\n    model = KMeans(n_clusters=2, random_state=42, n_init=10)\n    labels = model.fit_predict(X)\n    # TODO: Calculate silhouette_score(X, labels)\n    pass",
     "score = silhouette()\nassert -1 <= score <= 1"),
    
    ("Hierarchical Clustering", "Dendrogram", "Advanced",
     "def hierarchical():\n    from sklearn.cluster import AgglomerativeClustering\n    X = np.array([[1,2],[2,3],[8,9],[9,10]])\n    # TODO: Fit AgglomerativeClustering with n_clusters=2\n    pass",
     "model = hierarchical()\nassert hasattr(model,