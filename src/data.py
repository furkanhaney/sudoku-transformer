import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class SudokuDataset(Dataset):
    """
    Sudoku dataset that loads all data into memory and performs relabeling augmentation.

    Relabeling: Randomly permutes the digits 1-9 while preserving sudoku structure.
    For example: all 1s become 5s, all 2s become 7s, etc.
    """

    def __init__(self, puzzles: np.ndarray, solutions: np.ndarray, augment: bool = False):
        """
        Args:
            puzzles: numpy array of shape (N, 81) with values 0-9
            solutions: numpy array of shape (N, 81) with values 1-9
            augment: whether to apply relabeling augmentation
        """
        self.puzzles = torch.from_numpy(puzzles).long()
        self.solutions = torch.from_numpy(solutions).long()
        self.augment = augment

        assert len(self.puzzles) == len(self.solutions)
        assert self.puzzles.shape[1] == 81
        assert self.solutions.shape[1] == 81

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        puzzle = self.puzzles[idx].clone()
        solution = self.solutions[idx].clone()

        if self.augment:
            # Create random permutation of digits 1-9
            perm = torch.randperm(9) + 1  # [1-9] in random order

            # Create mapping from old digit to new digit
            # mapping[i] = new value for digit i
            mapping = torch.zeros(10, dtype=torch.long)
            mapping[0] = 0  # Keep 0 (empty cells) as 0
            mapping[1:] = perm

            # Apply relabeling
            puzzle = mapping[puzzle]
            solution = mapping[solution]

        # Convert solution from 1-9 to 0-8 for classification labels
        solution = solution - 1

        return puzzle, solution


def load_data(data_path: str | Path | None = None):
    """
    Load sudoku data from NPZ file.

    Args:
        data_path: Path to sudoku.npz file. If None, uses default path.

    Returns:
        Dictionary containing train_x, train_y, valid_x, valid_y as numpy arrays
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "sudoku.npz"

    data = np.load(data_path)
    return {
        'train_x': data['train_x'],
        'train_y': data['train_y'],
        'valid_x': data['valid_x'],
        'valid_y': data['valid_y']
    }


def create_dataloaders(
    batch_size: int = 64,
    num_workers: int = 4,
    data_path: str | Path | None = None
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        data_path: Path to sudoku.npz file. If None, uses default path.

    Returns:
        Tuple of (train_loader, valid_loader)
    """
    # Load data
    data = load_data(data_path)

    # Create datasets
    train_dataset = SudokuDataset(
        data['train_x'],
        data['train_y'],
        augment=True  # Apply augmentation to training set
    )

    valid_dataset = SudokuDataset(
        data['valid_x'],
        data['valid_y'],
        augment=False  # No augmentation for validation set
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, valid_loader


if __name__ == "__main__":
    # Test the data loading
    print("Loading data...")
    train_loader, valid_loader = create_dataloaders(batch_size=32, num_workers=0)

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(valid_loader)}")

    # Test a batch
    puzzles, solutions = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Puzzles: {puzzles.shape}, dtype: {puzzles.dtype}")
    print(f"  Solutions: {solutions.shape}, dtype: {solutions.dtype}")
    print(f"\nValue ranges:")
    print(f"  Puzzles: {puzzles.min()}-{puzzles.max()}")
    print(f"  Solutions: {solutions.min()}-{solutions.max()} (targets are 0-8 for classification)")

    print(f"\nSample puzzle (first 9 cells):")
    print(f"  {puzzles[0, :9].tolist()}")
    print(f"Sample solution (first 9 cells):")
    print(f"  {solutions[0, :9].tolist()}")
