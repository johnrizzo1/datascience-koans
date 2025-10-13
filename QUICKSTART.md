# Quick Start Guide - Data Science Koans

Get started with Data Science Koans in 5 minutes! 🚀

## Prerequisites

- Python 3.8 or higher
- Basic Python knowledge
- 5 minutes of your time

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/datascience-koans.git
cd datascience-koans
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Jupyter
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- And more!

## Testing the Installation

Run the test script to verify everything works:

```bash
cd tests
python test_core_infrastructure.py
```

You should see:
```
🧪 TESTING DATA SCIENCE KOANS CORE INFRASTRUCTURE
============================================================
Testing KoanValidator...
Testing ProgressTracker...
Testing DataGenerator...
🎉 ALL TESTS PASSED!
```

## Your First Koan

### 1. Launch Jupyter

```bash
jupyter notebook
```

### 2. Navigate to Notebooks

In Jupyter, navigate to:
```
koans/notebooks/
```

### 3. Open First Notebook

Open `01_numpy_fundamentals.ipynb`

### 4. Run Setup Cell

Execute the first cell to initialize the validator and tracker.

### 5. Complete Your First Koan

1. Read the explanation for Koan 1
2. Complete the TODO section
3. Run the validation cell
4. Celebrate when you see ✅!

## Example Koan

Here's what a koan looks like:

```python
# === KOAN 1: Array Creation ===
# 🎯 Objective: Learn to create NumPy arrays
# 📊 Difficulty: Beginner

"""
NumPy arrays are the foundation of numerical computing in Python.
Create them using np.array() from Python lists.
"""

# TODO: Create a NumPy array containing the numbers 1 through 5
def create_simple_array():
    # Your code here
    return np.array([1, 2, 3, 4, 5])

# Validation
@validator.koan(1, "Array Creation", difficulty="Beginner")
def validate():
    result = create_simple_array()
    assert isinstance(result, np.ndarray), "Must return a NumPy array"
    assert result.shape == (5,), "Array should have 5 elements"
    
validate()
```

When you run the validation cell, you'll see:
```
✅ Koan 1: Array Creation - PASSED
   🎉 Great work! Moving forward...
```

## Checking Your Progress

At any time, run this in a notebook cell:

```python
tracker.display_progress()
```

You'll see:
- Overall completion percentage
- Mastery levels by topic
- Individual notebook progress

## Tips for Success

1. **Read Carefully** - Understand the concept before coding
2. **Start Simple** - Get something working, then improve
3. **Use Hints** - Error messages guide you to the solution
4. **Don't Rush** - Understanding > Speed
5. **Have Fun!** - Learning should be enjoyable 🎉

## Common Issues

### Import Errors

If you see import errors, make sure you're in the right directory:

```python
import sys
sys.path.append('../..')  # This should be in the setup cell
```

### Progress Not Saving

Progress is saved to `data/progress.json`. If it's not saving:
1. Check file permissions
2. Ensure the `data/` directory exists
3. Try running from the project root

### Validation Not Working

Make sure you:
1. Ran the setup cell first
2. Defined your function correctly
3. Are returning a value (not just printing)

## Next Steps

After completing the first notebook:

1. **Continue Learning** - Move to `02_pandas_essentials.ipynb`
2. **Track Progress** - Use `tracker.display_progress()`
3. **Review Concepts** - Revisit earlier koans if needed
4. **Share Feedback** - Help us improve!

## Getting Help

- 📖 **README.md** - Comprehensive project documentation
- 🐛 **GitHub Issues** - Report bugs or ask questions
- 💬 **Discussions** - Join the community
- 📧 **Email** - [Your contact]

## What's Next?

The project currently has:
- ✅ Complete core infrastructure
- ✅ First demo notebook (5 koans)
- 🔄 More content coming soon!

Full content roadmap:
- Level 1: Foundation (30 koans)
- Level 2: Data Preparation (30 koans)
- Level 3: Model Fundamentals (30 koans)
- Level 4: Advanced Techniques (30 koans)
- Level 5: Best Practices (10 koans)

**Total: 130 koans** across 15 notebooks

---

**Happy Learning!** 🧘‍♂️📊

*"The path to data science mastery is through practice, not just theory."*