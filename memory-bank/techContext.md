# Technical Context

## Technology Stack

### Core Languages & Frameworks
- **Python 3.8+**: Primary language
- **Jupyter Notebook/Lab**: Development and learning environment
- **IPython**: Enhanced Python shell for notebooks

### Data Science Libraries
- **NumPy 1.21+**: Numerical computing foundation
- **pandas 1.3+**: Data manipulation and analysis
- **scikit-learn 1.0+**: Machine learning algorithms and tools
- **matplotlib 3.4+**: Data visualization
- **seaborn 0.11+**: Statistical data visualization

### Testing & Validation
- **Custom validation framework**: Built on Python assertions
- **pytest** (optional): For testing the validation framework itself
- **unittest**: Built-in testing support

### Supporting Libraries
- **scipy**: Scientific computing utilities
- **joblib**: Serialization for sklearn objects
- **json**: Progress data persistence

## Development Setup

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
# or
jupyter lab
```

### Required Packages (requirements.txt)
```
jupyter>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

### Optional Development Packages
```
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0
ipywidgets>=7.6.0  # For interactive progress widgets
```

## Technical Constraints

### Performance Considerations
- Datasets kept reasonably small (< 100K rows) for responsiveness
- Validation should complete in < 1 second per koan
- Progress tracking must be lightweight (< 100ms)

### Compatibility Requirements
- Python 3.8+ (for walrus operator, positional-only parameters)
- Cross-platform (Windows, macOS, Linux)
- Works in both Jupyter Notebook and Jupyter Lab
- No GPU requirements (CPU-only operations)

### Storage Constraints
- Progress data stored locally as JSON
- No database requirements
- Minimal disk space (< 100MB total)

## Tool Usage Patterns

### Jupyter Notebook Best Practices
```python
# Always import at top of notebook
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Use magic commands for convenience
%matplotlib inline
%load_ext autoreload
%autoreload 2

# Clear output for clean re-runs
from IPython.display import clear_output
```

### Validation Pattern
```python
# Standard validation imports in each notebook
from koans.core.validator import KoanValidator
from koans.core.progress import ProgressTracker

# Initialize validator
validator = KoanValidator(notebook_name="01_numpy_fundamentals")

# Validate individual koan
@validator.koan(1, "Array Creation")
def test_array_creation():
    result = user_solution()
    expected = np.array([1, 2, 3])
    np.testing.assert_array_equal(result, expected)
```

### Data Generation Pattern
```python
from koans.core.data_gen import DataGenerator

# Generate data for specific koan
data = DataGenerator.for_regression(
    n_samples=100,
    n_features=5,
    noise=0.1,
    random_state=42
)
```

### Progress Tracking Pattern
```python
from koans.core.progress import ProgressTracker

tracker = ProgressTracker()
tracker.complete_koan(notebook_id="01", koan_id=1, score=1.0)
tracker.display_progress()  # Shows visual progress
tracker.get_mastery_report()  # Detailed mastery by topic
```

## Integration Patterns

### Notebook Initialization Block
```python
# Standard setup cell (first cell of every notebook)
import sys
sys.path.append('..')

from koans.core.validator import KoanValidator
from koans.core.progress import ProgressTracker
from koans.core.data_gen import DataGenerator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# Initialize notebook tracking
notebook_id = "01_numpy_fundamentals"
validator = KoanValidator(notebook_id)
tracker = ProgressTracker()

print(f"📚 Welcome to {notebook_id}")
print(f"📊 Your progress: {tracker.get_notebook_progress(notebook_id)}%")
```

### Koan Template Cell
```python
# === KOAN X: Title ===
# Learning Objective: What you'll learn
# Difficulty: Beginner/Intermediate/Advanced

"""
Conceptual explanation goes here.
This is where we explain the concept being tested.
"""

# TODO: Your task description here
def my_solution():
    # Your code here
    pass

# Validation (run this cell after completing TODO)
@validator.koan(X, "Title")
def validate():
    result = my_solution()
    # Assertions here
    assert result is not None, "Solution must return a value"
    
validate()
```

## Version Control Strategy
- Git for source control
- `.gitignore` includes:
  - `data/progress.json` (user-specific)
  - `.ipynb_checkpoints/`
  - `__pycache__/`
  - `venv/`
- Notebooks stored with cleared output
- Solutions in separate tracked file

## Deployment/Distribution
- GitHub repository for source code
- pip-installable package (optional)
- Clone and run locally (primary method)
- Possible future: Binder/Colab integration

## Testing Strategy
- Unit tests for validation framework
- Integration tests for progress tracking
- Example koans with known solutions
- Manual testing of user experience flow