# Data Science Koans 🧘‍♂️📊

Learn data science through practice, inspired by [Ruby Koans](https://www.rubykoans.com/).

## 🎯 What are Data Science Koans?

Data Science Koans is an interactive learning platform that teaches data science fundamentals through hands-on exercises. Each "koan" is a small exercise that tests your understanding of a specific concept. Complete them sequentially to build your data science skills from the ground up.

### Key Features

- ✅ **130 Progressive Exercises** - From NumPy basics to advanced ML pipelines
- 📊 **Immediate Feedback** - Know instantly if your solution is correct
- 📈 **Progress Tracking** - See your mastery levels across all topics
- 🎓 **Self-Paced Learning** - Work at your own speed
- 🔄 **Extensible Design** - Easy to add new koans and topics

## 📚 What You'll Learn

### Level 1: Foundation (30 koans)

- NumPy array operations and broadcasting
- Pandas DataFrames and data manipulation
- Basic data exploration techniques

### Level 2: Data Preparation (30 koans)

- Data cleaning and quality assessment
- Feature scaling and encoding
- Feature engineering fundamentals

### Level 3: Model Fundamentals (30 koans)

- Regression and classification basics
- Model evaluation techniques
- Understanding bias-variance tradeoff

### Level 4: Advanced Techniques (30 koans)

- Clustering and dimensionality reduction
- Ensemble methods (Random Forests, Boosting)
- Hyperparameter tuning strategies

### Level 5: Best Practices (10 koans)

- ML pipelines and workflows
- Ethics and fairness in machine learning
- Responsible data science practices

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Basic Python knowledge
- Enthusiasm for learning! 🎉

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/datascience-koans.git
cd datascience-koans
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Launch Jupyter**

```bash
jupyter notebook
```

5. **Start with the first notebook**

Navigate to `koans/notebooks/` and open `01_numpy_fundamentals.ipynb`

## 📖 How to Use

Each notebook contains multiple koans following this structure:

```python
# === KOAN X: Title ===
# 🎯 Objective: What you'll learn
# 📊 Difficulty: Beginner/Intermediate/Advanced

"""
Explanation of the concept you're about to practice.
"""

# TODO: Complete this function
def my_solution():
    # Your code here
    pass

# Validation - run this cell after completing the TODO
@validator.koan(X, "Title", difficulty="Beginner")
def validate():
    result = my_solution()
    assert result is not None, "Must return a value"
    
validate()
```

### Workflow

1. **Read** the koan explanation
2. **Write** your solution in the TODO section
3. **Run** the validation cell
4. **Iterate** if needed based on feedback
5. **Celebrate** success! ✅
6. **Move** to the next koan

### Checking Progress

At any time, you can check your progress:

```python
from koans.core.progress import ProgressTracker

tracker = ProgressTracker()
tracker.display_progress()
```

This shows:

- Overall completion percentage
- Mastery levels by topic
- Individual notebook progress

## 📂 Project Structure

```
datascience-koans/
├── README.md                      # This file
├── requirements.txt               # Dependencies
├── setup.py                       # Package setup
├── koans/                         # Main package
│   ├── core/                      # Core infrastructure
│   │   ├── validator.py           # Validation framework
│   │   ├── progress.py            # Progress tracking
│   │   └── data_gen.py            # Data generation
│   ├── notebooks/                 # Learning notebooks (15 total)
│   │   ├── 01_numpy_fundamentals.ipynb
│   │   ├── 02_pandas_essentials.ipynb
│   │   └── ...
│   └── solutions/                 # Reference solutions
├── data/                          # Your progress data
└── tests/                         # Test suite
```

## 🎓 Learning Path

We recommend following the notebooks in order:

1. **Start with Level 1** - Build a solid foundation
2. **Progress sequentially** - Each level builds on the previous
3. **Complete all koans** in a notebook before moving on
4. **Review as needed** - Revisit earlier koans if concepts are unclear
5. **Check progress regularly** - Celebrate your achievements!

## 💡 Tips for Success

- ✨ **Read explanations carefully** - Understanding "why" is as important as "how"
- 🔄 **Don't fear failure** - Mistakes are learning opportunities
- 📝 **Take notes** - Document insights in markdown cells
- 🤔 **Think before looking** - Try to solve it yourself first
- 💬 **Ask for help** - Use the solutions as a last resort
- 🎯 **Practice regularly** - Consistent practice builds mastery

## 🤝 Contributing

We welcome contributions! Ways you can help:

- 📝 Add new koans to existing notebooks
- 📚 Create new topic notebooks
- 🐛 Fix bugs or improve code
- 📖 Improve documentation
- ✨ Enhance validation messages

See `CONTRIBUTING.md` for guidelines (coming soon).

## 📊 Mastery Levels

Your mastery in each topic is calculated based on completion:

- 🏆 **Master** (90-100%) - You've got this down!
- ⭐ **Proficient** (70-89%) - Strong understanding
- 📚 **Learning** (50-69%) - Making good progress
- 🌱 **Beginner** (0-49%) - Just getting started

## 🔮 Future Enhancements

Planned additions:

- Deep learning basics (TensorFlow/PyTorch)
- Natural language processing koans
- Time series analysis
- Computer vision fundamentals
- Interactive hints system
- Web-based interface

## 📜 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Inspired by [Ruby Koans](https://www.rubykoans.com/) by EdgeCase
- Built with love for the data science community
- Special thanks to all contributors

## 📞 Support

- 🐛 **Issues**: Report bugs via GitHub Issues
- 💬 **Discussions**: Join our community discussions
- 📧 **Contact**: [Your contact info]

---

**Happy Learning!** 🚀📊🧘‍♂️

*"The path to data science mastery is through practice, not just theory."*
