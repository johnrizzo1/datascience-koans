"""
Validation framework for Data Science Koans.

Provides decorator-based validation with immediate feedback and automatic
progress tracking integration.
"""

from typing import Callable, Optional, Dict, Any
from functools import wraps


class KoanValidator:
    """
    Validates koan solutions and provides immediate feedback.
    
    The validator uses a decorator pattern to wrap validation functions,
    providing consistent feedback formatting and automatic progress tracking.
    
    Example:
        validator = KoanValidator("01_numpy_fundamentals")
        
        @validator.koan(1, "Array Creation", difficulty="Beginner")
        def validate():
            result = my_solution()
            assert result is not None, "Must return a value"
            
        validate()
    """
    
    def __init__(self, notebook_id: str):
        """
        Initialize validator for a specific notebook.
        
        Args:
            notebook_id: Unique identifier for the notebook (e.g., "01_numpy_fundamentals")
        """
        self.notebook_id = notebook_id
        self.koans: Dict[int, Dict[str, Any]] = {}
        self.results: Dict[int, Dict[str, Any]] = {}
    
    def koan(self, 
             koan_number: int, 
             title: str, 
             difficulty: str = "Beginner") -> Callable:
        """
        Decorator for koan validation functions.
        
        Wraps a validation function to provide:
        - Consistent success/failure formatting
        - Helpful error messages
        - Automatic progress tracking
        
        Args:
            koan_number: Sequential number of the koan
            title: Short descriptive title
            difficulty: One of "Beginner", "Intermediate", "Advanced"
            
        Returns:
            Decorated validation function
            
        Example:
            @validator.koan(1, "Array Creation", difficulty="Beginner")
            def validate():
                result = create_array()
                assert isinstance(result, np.ndarray), "Must return ndarray"
        """
        def decorator(func: Callable) -> Callable:
            # Store koan metadata
            self.koans[koan_number] = {
                'title': title,
                'difficulty': difficulty,
                'validator': func
            }
            
            @wraps(func)
            def wrapper(*args, **kwargs) -> bool:
                """Execute validation and provide feedback"""
                try:
                    # Run the validation function
                    func(*args, **kwargs)
                    self._mark_success(koan_number)
                    return True
                    
                except AssertionError as e:
                    self._mark_failure(koan_number, str(e))
                    return False
                    
                except Exception as e:
                    self._mark_error(koan_number, str(e))
                    return False
            
            return wrapper
        return decorator
    
    def _mark_success(self, koan_number: int) -> None:
        """
        Record successful koan completion.
        
        Displays success message and updates progress tracker.
        """
        koan = self.koans[koan_number]
        print(f"✅ Koan {koan_number}: {koan['title']} - PASSED")
        print(f"   🎉 Great work! Moving forward...\n")
        
        # Record result
        self.results[koan_number] = {
            'status': 'passed',
            'score': 1.0
        }
        
        # Update progress tracker
        try:
            from koans.core.progress import ProgressTracker
            tracker = ProgressTracker()
            tracker.complete_koan(self.notebook_id, koan_number, 1.0)
        except Exception as e:
            # Don't fail validation if progress tracking fails
            print(f"   ⚠️  Warning: Could not update progress: {e}")
    
    def _mark_failure(self, koan_number: int, error_msg: str) -> None:
        """
        Record failed koan attempt.
        
        Displays failure message with helpful hints.
        """
        koan = self.koans[koan_number]
        print(f"❌ Koan {koan_number}: {koan['title']} - FAILED")
        print(f"   Error: {error_msg}")
        print(f"   💡 Hint: Review the concept explanation and try again!")
        print(f"   📚 Check your understanding of the fundamentals.\n")
        
        # Record result
        self.results[koan_number] = {
            'status': 'failed',
            'error': error_msg
        }
    
    def _mark_error(self, koan_number: int, error_msg: str) -> None:
        """
        Record koan execution error.
        
        Displays error message for syntax or runtime errors.
        """
        koan = self.koans[koan_number]
        print(f"⚠️  Koan {koan_number}: {koan['title']} - ERROR")
        print(f"   Error: {error_msg}")
        print(f"   🔍 Check your code for syntax or runtime errors.")
        print(f"   💭 Make sure all required variables and functions are defined.\n")
        
        # Record result
        self.results[koan_number] = {
            'status': 'error',
            'error': error_msg
        }
    
    def get_results(self) -> Dict[int, Dict[str, Any]]:
        """
        Get all validation results for this notebook.
        
        Returns:
            Dictionary mapping koan numbers to their results
        """
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for this notebook.
        
        Returns:
            Dictionary with counts of passed/failed/error koans
        """
        passed = sum(1 for r in self.results.values() if r['status'] == 'passed')
        failed = sum(1 for r in self.results.values() if r['status'] == 'failed')
        errors = sum(1 for r in self.results.values() if r['status'] == 'error')
        
        return {
            'total_koans': len(self.koans),
            'attempted': len(self.results),
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'completion_rate': (passed / len(self.koans) * 100) if self.koans else 0
        }
    
    def display_summary(self) -> None:
        """Display a formatted summary of validation results."""
        summary = self.get_summary()
        
        print("=" * 60)
        print(f"📊 Koan Summary for {self.notebook_id}")
        print("=" * 60)
        print(f"Total Koans:       {summary['total_koans']}")
        print(f"Attempted:         {summary['attempted']}")
        print(f"✅ Passed:         {summary['passed']}")
        print(f"❌ Failed:         {summary['failed']}")
        print(f"⚠️  Errors:         {summary['errors']}")
        print(f"Completion Rate:   {summary['completion_rate']:.1f}%")
        print("=" * 60)