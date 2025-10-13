"""
Data Science Koans - Learn data science through practice

A collection of interactive exercises teaching NumPy, pandas, and scikit-learn
fundamentals through hands-on practice with immediate feedback.

Inspired by Ruby Koans: https://www.rubykoans.com/
"""

__version__ = "0.1.0"
__author__ = "Data Science Koans Contributors"

from koans.core.validator import KoanValidator
from koans.core.progress import ProgressTracker
from koans.core.data_gen import DataGenerator

__all__ = ['KoanValidator', 'ProgressTracker', 'DataGenerator']