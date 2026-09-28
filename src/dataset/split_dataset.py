"""
Dataset Split Script (Person 4).
Generates speaker-independent train / validation / test splits:
- Ensures speakers in test/val sets are never seen during training
- Creates balanced representation across all 10 digit classes (d0–d9)
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def create_speaker_independent_split(
    metadata_df: pd.DataFrame,
    train_speakers: List[str],
    val_speakers: List[str],
    test_speakers: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Split metadata by speaker IDs to enforce speaker independence.

    Args:
        metadata_df: Master metadata DataFrame.
        train_speakers: List of speaker IDs for training (e.g. 20 speakers).
        val_speakers: List of speaker IDs for validation (e.g. 5 speakers).
        test_speakers: List of speaker IDs for testing (e.g. 5 speakers).

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to split DataFrames.
    """
    # Skeleton placeholder
    pass


if __name__ == "__main__":
    print("Split dataset module skeleton initialized.")
