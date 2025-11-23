import numpy as np
import pandas as pd
from pathlib import Path


def process_sudoku_data():
    """
    Process sudoku.csv and convert to sudoku.npz with train/validation splits.

    Creates:
        - train_x: (990_000, 81) puzzles with values 0-9
        - train_y: (990_000, 81) solutions with values 1-9
        - valid_x: (10_000, 81) puzzles with values 0-9
        - valid_y: (10_000, 81) solutions with values 1-9
    All arrays are np.uint8 dtype.
    """
    # Define paths
    data_dir = Path(__file__).parent.parent / "data"
    csv_path = data_dir / "sudoku.csv"
    npz_path = data_dir / "sudoku.npz"

    print(f"Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} puzzles")

    # Convert string puzzles to numpy arrays
    print("Converting puzzles to arrays...")
    puzzles = np.array([[int(c) for c in quiz] for quiz in df['quizzes']], dtype=np.uint8)
    solutions = np.array([[int(c) for c in sol] for sol in df['solutions']], dtype=np.uint8)

    # Split into train and validation
    # Last 10,000 for validation, rest for training
    n_valid = 10_000

    valid_x = puzzles[-n_valid:]
    valid_y = solutions[-n_valid:]
    train_x = puzzles[:-n_valid]
    train_y = solutions[:-n_valid]

    print(f"Training set: {train_x.shape}")
    print(f"Validation set: {valid_x.shape}")
    print(f"Puzzle values range: {puzzles.min()}-{puzzles.max()}")
    print(f"Solution values range: {solutions.min()}-{solutions.max()}")

    # Verify shapes and dtypes
    assert train_x.shape == (990_000, 81), f"Unexpected train_x shape: {train_x.shape}"
    assert train_y.shape == (990_000, 81), f"Unexpected train_y shape: {train_y.shape}"
    assert valid_x.shape == (10_000, 81), f"Unexpected valid_x shape: {valid_x.shape}"
    assert valid_y.shape == (10_000, 81), f"Unexpected valid_y shape: {valid_y.shape}"
    assert train_x.dtype == np.uint8
    assert train_y.dtype == np.uint8
    assert valid_x.dtype == np.uint8
    assert valid_y.dtype == np.uint8

    # Save to NPZ
    print(f"Saving to {npz_path}...")
    np.savez_compressed(
        npz_path,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y
    )

    print("Done!")


if __name__ == "__main__":
    process_sudoku_data()
