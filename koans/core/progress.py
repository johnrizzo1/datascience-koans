"""
Progress tracking system for Data Science Koans.

Tracks learner progress across all notebooks, calculates mastery levels,
and provides visual progress reporting.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class ProgressTracker:
    """
    Tracks learner progress and calculates mastery levels.
    
    Progress is persisted to a JSON file, allowing learners to resume
    their journey across sessions. Mastery levels are calculated per topic
    based on completion rates and difficulty weights.
    
    Example:
        tracker = ProgressTracker()
        tracker.complete_koan("01_numpy_fundamentals", 1, 1.0)
        tracker.display_progress()
        report = tracker.get_mastery_report()
    """
    
    # Topic mappings for mastery calculation
    TOPIC_NOTEBOOKS = {
        'numpy': ['01_numpy_fundamentals'],
        'pandas': ['02_pandas_essentials', '03_data_exploration'],
        'preprocessing': [
            '04_data_cleaning',
            '05_data_transformation',
            '06_feature_engineering_basics'
        ],
        'regression': ['07_regression_basics'],
        'classification': ['08_classification_basics'],
        'evaluation': ['09_model_evaluation'],
        'clustering': ['10_clustering'],
        'dimensionality_reduction': ['11_dimensionality_reduction'],
        'ensemble': ['12_ensemble_methods'],
        'tuning': ['13_hyperparameter_tuning'],
        'pipelines': ['14_model_selection_pipeline'],
        'ethics': ['15_ethics_and_bias']
    }
    
    # Expected koan counts per notebook
    NOTEBOOK_KOAN_COUNTS = {
        '01_numpy_fundamentals': 10,
        '02_pandas_essentials': 10,
        '03_data_exploration': 10,
        '04_data_cleaning': 10,
        '05_data_transformation': 10,
        '06_feature_engineering_basics': 10,
        '07_regression_basics': 10,
        '08_classification_basics': 10,
        '09_model_evaluation': 10,
        '10_clustering': 8,
        '11_dimensionality_reduction': 8,
        '12_ensemble_methods': 7,
        '13_hyperparameter_tuning': 7,
        '14_model_selection_pipeline': 5,
        '15_ethics_and_bias': 5
    }
    
    def __init__(self, progress_file: str = "data/progress.json"):
        """
        Initialize progress tracker.
        
        Args:
            progress_file: Path to JSON file for storing progress
        """
        self.progress_file = Path(progress_file)
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load_progress()
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load progress from JSON file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # If file is corrupted, start fresh
                return self._init_progress_data()
        return self._init_progress_data()
    
    def _init_progress_data(self) -> Dict[str, Any]:
        """Initialize empty progress data structure."""
        return {
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'notebooks': {},
            'total_koans_completed': 0,
            'mastery_levels': {}
        }
    
    def _save_progress(self) -> None:
        """Save progress to JSON file."""
        self.data['last_updated'] = datetime.now().isoformat()
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save progress: {e}")
    
    def complete_koan(self,
                      notebook_id: str,
                      koan_number: int,
                      score: float = 1.0) -> None:
        """
        Mark a koan as completed.
        
        Args:
            notebook_id: Notebook identifier
            koan_number: Sequential koan number within notebook
            score: Score for the koan (0.0 to 1.0)
        """
        # Initialize notebook if not exists
        if notebook_id not in self.data['notebooks']:
            self.data['notebooks'][notebook_id] = {
                'koans': {},
                'started_at': datetime.now().isoformat()
            }
        
        # Record koan completion
        self.data['notebooks'][notebook_id]['koans'][str(koan_number)] = {
            'completed_at': datetime.now().isoformat(),
            'score': score
        }
        
        # Update total count
        self.data['total_koans_completed'] = sum(
            len(nb['koans'])
            for nb in self.data['notebooks'].values()
        )
        
        # Recalculate mastery levels
        self._update_mastery_levels()
        
        # Save to disk
        self._save_progress()
    
    def _update_mastery_levels(self) -> None:
        """Calculate mastery levels for all topics."""
        for topic, notebooks in self.TOPIC_NOTEBOOKS.items():
            self.data['mastery_levels'][topic] = self._calc_mastery(
                notebooks
            )
    
    def _calc_mastery(self, notebook_ids: List[str]) -> float:
        """
        Calculate mastery percentage for a set of notebooks.
        
        Args:
            notebook_ids: List of notebook identifiers
            
        Returns:
            Mastery percentage (0-100)
        """
        total_koans = sum(
            self.NOTEBOOK_KOAN_COUNTS.get(nb_id, 0)
            for nb_id in notebook_ids
        )
        
        if total_koans == 0:
            return 0.0
        
        completed_koans = sum(
            len(self.data['notebooks'].get(nb_id, {}).get('koans', {}))
            for nb_id in notebook_ids
        )
        
        return (completed_koans / total_koans) * 100
    
    def get_notebook_progress(self, notebook_id: str) -> int:
        """
        Get completion percentage for a specific notebook.
        
        Args:
            notebook_id: Notebook identifier
            
        Returns:
            Completion percentage (0-100)
        """
        expected = self.NOTEBOOK_KOAN_COUNTS.get(notebook_id, 0)
        if expected == 0:
            return 0
        
        completed = len(
            self.data['notebooks'].get(notebook_id, {}).get('koans', {})
        )
        return int((completed / expected) * 100)
    
    def get_mastery_report(self) -> Dict[str, float]:
        """
        Get mastery levels for all topics.
        
        Returns:
            Dictionary mapping topic names to mastery percentages
        """
        return self.data.get('mastery_levels', {})
    
    def get_mastery_level_label(self, percentage: float) -> str:
        """
        Convert mastery percentage to a label.
        
        Args:
            percentage: Mastery percentage (0-100)
            
        Returns:
            Label: Master, Proficient, Learning, or Beginner
        """
        if percentage >= 90:
            return "🏆 Master"
        elif percentage >= 70:
            return "⭐ Proficient"
        elif percentage >= 50:
            return "📚 Learning"
        else:
            return
            return "🌱 Beginner"
    
    def display_progress(self) -> None:
        """Display formatted progress report."""
        print("\n" + "=" * 70)
        print("📊 YOUR DATA SCIENCE KOANS PROGRESS")
        print("=" * 70)
        
        # Overall stats
        total = sum(self.NOTEBOOK_KOAN_COUNTS.values())
        completed = self.data['total_koans_completed']
        overall_pct = (completed / total * 100) if total > 0 else 0
        
        print(f"\n🎯 Overall Progress: {completed}/{total} koans "
              f"({overall_pct:.1f}%)")
        
        # Mastery by topic
        print("\n📚 Mastery by Topic:")
        print("-" * 70)
        
        mastery = self.get_mastery_report()
        if mastery:
            for topic, pct in sorted(mastery.items()):
                label = self.get_mastery_level_label(pct)
                bar = self._progress_bar(pct)
                topic_display = topic.replace('_', ' ').title()
                print(f"  {topic_display:25} {bar} {pct:5.1f}% {label}")
        else:
            print("  No progress yet. Start with notebook 01!")
        
        # Notebook completion
        print("\n📓 Notebook Completion:")
        print("-" * 70)
        
        for nb_id in sorted(self.NOTEBOOK_KOAN_COUNTS.keys()):
            pct = self.get_notebook_progress(nb_id)
            bar = self._progress_bar(pct)
            status = "✓" if pct == 100 else " "
            print(f"  {status} {nb_id:30} {bar} {pct:3d}%")
        
        print("=" * 70 + "\n")
    
    def _progress_bar(self, percentage: float, width: int = 20) -> str:
        """
        Create a text progress bar.
        
        Args:
            percentage: Progress percentage (0-100)
            width: Width of the bar in characters
            
        Returns:
            String representation of progress bar
        """
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    def get_next_recommendation(self) -> Optional[str]:
        """
        Recommend the next notebook to work on.
        
        Returns:
            Notebook ID or None if all complete
        """
        for nb_id in sorted(self.NOTEBOOK_KOAN_COUNTS.keys()):
            if self.get_notebook_progress(nb_id) < 100:
                return nb_id
        return None
    
    def reset_progress(self) -> None:
        """Reset all progress data."""
        self.data = self._init_progress_data()
        self._save_progress()
        print("✨ Progress has been reset!")
    
    def export_progress(self, filepath: str) -> None:
        """
        Export progress to a file.
        
        Args:
            filepath: Path to export file
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.data, f, indent=2)
            print(f"✅ Progress exported to {filepath}")
        except IOError as e:
            print(f"❌ Could not export progress: {e}")