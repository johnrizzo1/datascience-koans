"""
Data generation utilities for Data Science Koans.

Provides functions to generate synthetic datasets and load real datasets
for use in koan exercises.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn import datasets


class DataGenerator:
    """
    Generates synthetic and real datasets for koan exercises.
    
    Provides consistent, reproducible datasets with specified properties
    to support isolated concept learning.
    
    Example:
        # Generate regression data
        X, y = DataGenerator.for_regression(n_samples=100, n_features=5)
        
        # Generate classification data
        X, y = DataGenerator.for_classification(n_samples=200, n_classes=3)
        
        # Load sklearn dataset
        data = DataGenerator.load_sklearn_dataset('iris')
    """
    
    @staticmethod
    def for_regression(n_samples: int = 100,
                       n_features: int = 1,
                       noise: float = 0.1,
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic regression dataset.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            noise: Standard deviation of Gaussian noise
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X, y) arrays
        """
        return datasets.make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state
        )
    
    @staticmethod
    def for_classification(n_samples: int = 100,
                          n_features: int = 2,
                          n_classes: int = 2,
                          n_clusters_per_class: int = 1,
                          random_state: int = 42) -> Tuple[np.ndarray,
                                                            np.ndarray]:
        """
        Generate synthetic classification dataset.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes
            n_clusters_per_class: Number of clusters per class
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X, y) arrays
        """
        return datasets.make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class,
            n_redundant=0,
            n_informative=n_features,
            random_state=random_state
        )
    
    @staticmethod
    def for_clustering(n_samples: int = 300,
                      n_features: int = 2,
                      n_clusters: int = 3,
                      cluster_std: float = 1.0,
                      random_state: int = 42) -> Tuple[np.ndarray,
                                                        np.ndarray]:
        """
        Generate synthetic clustering dataset.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_clusters: Number of clusters
            cluster_std: Standard deviation of clusters
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X, y) arrays where y contains true cluster labels
        """
        return datasets.make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=cluster_std,
            random_state=random_state
        )
    
    @staticmethod
    def synthetic_tabular(n_rows: int = 100,
                         n_numeric: int = 3,
                         n_categorical: int = 2,
                         missing_rate: float = 0.0,
                         random_state: int = 42) -> pd.DataFrame:
        """
        Generate synthetic tabular dataset with mixed types.
        
        Args:
            n_rows: Number of rows
            n_numeric: Number of numeric columns
            n_categorical: Number of categorical columns
            missing_rate: Proportion of missing values (0.0 to 1.0)
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with mixed data types
        """
        np.random.seed(random_state)
        
        data = {}
        
        # Generate numeric columns
        for i in range(n_numeric):
            col_name = f'numeric_{i+1}'
            data[col_name] = np.random.randn(n_rows)
        
        # Generate categorical columns
        categories = ['A', 'B', 'C', 'D', 'E']
        for i in range(n_categorical):
            col_name = f'category_{i+1}'
            data[col_name] = np.random.choice(categories, n_rows)
        
        df = pd.DataFrame(data)
        
        # Add missing values if requested
        if missing_rate > 0:
            mask = np.random.random(df.shape) < missing_rate
            df = df.mask(mask)
        
        return df
    
    @staticmethod
    def load_sklearn_dataset(name: str) -> pd.DataFrame:
        """
        Load a scikit-learn dataset as DataFrame.
        
        Args:
            name: Dataset name ('iris', 'wine', 'breast_cancer', 'digits')
            
        Returns:
            DataFrame with features and 'target' column
            
        Raises:
            ValueError: If dataset name is not recognized
        """
        loaders = {
            'iris': datasets.load_iris,
            'wine': datasets.load_wine,
            'breast_cancer': datasets.load_breast_cancer,
            'digits': datasets.load_digits
        }
        
        if name not in loaders:
            raise ValueError(
                f"Unknown dataset: {name}. "
                f"Available: {list(loaders.keys())}"
            )
        
        dataset = loaders[name]()
        df = pd.DataFrame(
            dataset.data,
            columns=dataset.feature_names
        )
        df['target'] = dataset.target
        
        return df
    
    @staticmethod
    def time_series(n_points: int = 365,
                   trend: float = 0.5,
                   seasonality: int = 7,
                   noise: float = 1.0,
                   random_state: int = 42) -> pd.Series:
        """
        Generate synthetic time series data.
        
        Args:
            n_points: Number of time points
            trend: Trend coefficient
            seasonality: Period of seasonal component
            noise: Standard deviation of random noise
            random_state: Random seed for reproducibility
            
        Returns:
            Series with datetime index
        """
        np.random.seed(random_state)
        
        # Create time index
        dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
        
        # Generate components
        t = np.arange(n_points)
        trend_component = trend * t
        seasonal_component = 10 * np.sin(2 * np.pi * t / seasonality)
        noise_component = noise * np.random.randn(n_points)
        
        # Combine
        values = trend_component + seasonal_component + noise_component
        
        return pd.Series(values, index=dates, name='value')
    
    @staticmethod
    def with_outliers(data: np.ndarray,
                     outlier_fraction: float = 0.1,
                     outlier_magnitude: float = 3.0,
                     random_state: int = 42) -> np.ndarray:
        """
        Add outliers to existing data.
        
        Args:
            data: Input array
            outlier_fraction: Fraction of points to make outliers
            outlier_magnitude: How many standard deviations away
            random_state: Random see
d for reproducibility
            
        Returns:
            Array with outliers added
        """
        np.random.seed(random_state)
        
        data = data.copy()
        n_outliers = int(len(data) * outlier_fraction)
        outlier_indices = np.random.choice(
            len(data),
            n_outliers,
            replace=False
        )
        
        # Calculate mean and std
        mean = np.mean(data)
        std = np.std(data)
        
        # Add outliers
        data[outlier_indices] = mean + outlier_magnitude * std * \
            np.random.choice([-1, 1], n_outliers)
        
        return data
    
    @staticmethod
    def imbalanced_classification(n_samples: int = 1000,
                                  n_features: int = 10,
                                  weights: Tuple[float, ...] = (0.9, 0.1),
                                  random_state: int = 42) -> Tuple[np.ndarray,
                                                                    np.ndarray]:
        """
        Generate imbalanced classification dataset.
        
        Args:
            n_samples: Total number of samples
            n_features: Number of features
            weights: Class distribution weights (must sum to 1.0)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X, y) arrays
        """
        return datasets.make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=len(weights),
            n_informative=n_features,
            n_redundant=0,
            weights=list(weights),
            random_state=random_state
        )