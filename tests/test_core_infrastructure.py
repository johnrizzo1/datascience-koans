"""
Test script to verify core infrastructure is working.

Run this to ensure the validation framework, progress tracker,
and data generator are functioning correctly.
"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from koans.core.validator import KoanValidator
from koans.core.progress import ProgressTracker
from koans.core.data_gen import DataGenerator


def test_validator():
    """Test the KoanValidator"""
    print("\n" + "="*60)
    print("Testing KoanValidator...")
    print("="*60)
    
    validator = KoanValidator("test_notebook")
    
    # Test a passing koan
    @validator.koan(1, "Test Koan 1", difficulty="Beginner")
    def test_pass():
        result = 2 + 2
        assert result == 4, "Math should work"
    
    print("\n1. Testing passing koan:")
    success = test_pass()
    assert success == True, "Should return True for passing koan"
    
    # Test a failing koan
    @validator.koan(2, "Test Koan 2", difficulty="Beginner")
    def test_fail():
        result = 2 + 2
        assert result == 5, "This will fail"
    
    print("\n2. Testing failing koan:")
    success = test_fail()
    assert success == False, "Should return False for failing koan"
    
    # Test summary
    print("\n3. Testing summary:")
    validator.display_summary()
    summary = validator.get_summary()
    assert summary['total_koans'] == 2
    assert summary['passed'] == 1
    assert summary['failed'] == 1
    
    print("\n✅ KoanValidator tests passed!")


def test_progress_tracker():
    """Test the ProgressTracker"""
    print("\n" + "="*60)
    print("Testing ProgressTracker...")
    print("="*60)
    
    tracker = ProgressTracker("test_data/test_progress.json")
    
    # Test completing koans
    print("\n1. Testing koan completion:")
    tracker.complete_koan("01_numpy_fundamentals", 1, 1.0)
    tracker.complete_koan("01_numpy_fundamentals", 2, 1.0)
    
    # Test progress retrieval
    print("\n2. Testing progress retrieval:")
    progress = tracker.get_notebook_progress("01_numpy_fundamentals")
    print(f"   Progress: {progress}%")
    assert progress == 20, "Should be 20% (2 out of 10)"
    
    # Test mastery report
    print("\n3. Testing mastery report:")
    mastery = tracker.get_mastery_report()
    print(f"   Mastery levels: {mastery}")
    
    # Test display
    print("\n4. Testing progress display:")
    tracker.display_progress()
    
    print("\n✅ ProgressTracker tests passed!")


def test_data_generator():
    """Test the DataGenerator"""
    print("\n" + "="*60)
    print("Testing DataGenerator...")
    print("="*60)
    
    # Test regression data
    print("\n1. Testing regression data generation:")
    X, y = DataGenerator.for_regression(n_samples=100, n_features=5)
    assert X.shape == (100, 5), "X should have correct shape"
    assert y.shape == (100,), "y should have correct shape"
    print(f"   Generated regression data: X.shape={X.shape}, y.shape={y.shape}")
    
    # Test classification data
    print("\n2. Testing classification data generation:")
    X, y = DataGenerator.for_classification(n_samples=200, n_features=10, n_classes=3)
    assert X.shape == (200, 10), "X should have correct shape"
    assert len(np.unique(y)) == 3, "Should have 3 classes"
    print(f"   Generated classification data: X.shape={X.shape}, classes={len(np.unique(y))}")
    
    # Test clustering data
    print("\n3. Testing clustering data generation:")
    X, y = DataGenerator.for_clustering(n_samples=300, n_clusters=4)
    assert X.shape[0] == 300, "Should have correct number of samples"
    print(f"   Generated clustering data: X.shape={X.shape}")
    
    # Test synthetic tabular data
    print("\n4. Testing synthetic tabular data:")
    df = DataGenerator.synthetic_tabular(n_rows=50, n_numeric=3, n_categorical=2)
    assert isinstance(df, pd.DataFrame), "Should return DataFrame"
    assert len(df) == 50, "Should have correct number of rows"
    assert len(df.columns) == 5, "Should have correct number of columns"
    print(f"   Generated tabular data: shape={df.shape}")
    
    # Test sklearn dataset loading
    print("\n5. Testing sklearn dataset loading:")
    df = DataGenerator.load_sklearn_dataset('iris')
    assert isinstance(df, pd.DataFrame), "Should return DataFrame"
    assert 'target' in df.columns, "Should have target column"
    print(f"   Loaded iris dataset: shape={df.shape}")
    
    print("\n✅ DataGenerator tests passed!")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 TESTING DATA SCIENCE KOANS CORE INFRASTRUCTURE")
    print("="*60)
    
    try:
        test_validator()
        test_progress_tracker()
        test_data_generator()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nCore infrastructure is working correctly!")
        print("You can now start creating koans and notebooks.")
        print("\nNext steps:")
        print("  1. Open koans/notebooks/01_numpy_fundamentals.ipynb")
        print("  2. Try completing the koans")
        print("  3. Check your progress with tracker.display_progress()")
        print("\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()