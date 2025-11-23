import csv
import torch
import shutil
from time import time
from tqdm import trange
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import DataLoader

from utils import Metric, GPTConfig


def get_device():
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def compute_accuracy(predictions, targets, inputs):
    """
    Compute accuracy only on non-hint positions (where input == 0).

    Args:
        predictions: (B, 81, 9) logits
        targets: (B, 81) ground truth labels (0-8)
        inputs: (B, 81) puzzle inputs (0-9)

    Returns:
        Accuracy on non-hint positions only
    """
    # Get predicted classes: (B, 81)
    pred_classes = predictions.argmax(dim=-1)

    # Create mask for non-hint positions (where input == 0)
    non_hint_mask = inputs == 0

    # Calculate accuracy only on non-hint positions
    correct = (pred_classes == targets) & non_hint_mask
    total_non_hints = non_hint_mask.sum()

    if total_non_hints == 0:
        return 0.0

    accuracy = correct.sum().float() / total_non_hints.float()
    return accuracy.item()


def compute_full_accuracy(predictions, targets, inputs):
    """
    Compute per-puzzle accuracy (1 if all non-hint positions correct, 0 otherwise).
    Always masks hints regardless of mask_hints config setting.

    Args:
        predictions: (B, 81, 9) logits
        targets: (B, 81) ground truth labels (0-8)
        inputs: (B, 81) puzzle inputs (0-9)

    Returns:
        Fraction of puzzles where all non-hint positions are correct
    """
    # Get predicted classes: (B, 81)
    pred_classes = predictions.argmax(dim=-1)

    # Create mask for non-hint positions (where input == 0)
    non_hint_mask = inputs == 0

    # Check correctness only on non-hint positions
    correct_at_non_hints = (pred_classes == targets) | ~non_hint_mask

    # A puzzle is fully solved if all non-hint positions are correct
    all_correct = correct_at_non_hints.all(dim=1)

    # Return fraction of fully correct puzzles
    return all_correct.float().mean().item()


def compute_loss(criterion, predictions, targets, inputs, mask_hints=False):
    """
    Compute loss with optional hint masking.

    Args:
        criterion: Loss function (should have reduction='none')
        predictions: (B, 81, 9) logits
        targets: (B, 81) ground truth labels (0-8)
        inputs: (B, 81) puzzle inputs (0-9)
        mask_hints: If True, only compute loss on non-hint positions

    Returns:
        Scalar loss value
    """
    B, T, C = predictions.shape
    predictions_flat = predictions.view(B * T, C)
    targets_flat = targets.view(B * T)

    # Compute per-element loss
    loss = criterion(predictions_flat, targets_flat)

    if mask_hints:
        # Create mask for non-hint positions (where input == 0)
        inputs_flat = inputs.view(B * T)
        mask = (inputs_flat == 0).float()

        # Apply mask and compute mean only over non-hint positions
        masked_loss = loss * mask
        total_non_hints = mask.sum()

        if total_non_hints > 0:
            return masked_loss.sum() / total_non_hints
        else:
            return masked_loss.sum()  # Should not happen in practice
    else:
        # Regular mean over all positions
        return loss.mean()


def setup_experiment_dir(config: GPTConfig, config_path: str):
    """
    Setup experiment directory and copy config file.

    Args:
        config: Configuration object
        config_path: Path to config JSON file

    Returns:
        Path to experiment directory
    """
    exp_dir = Path("models") / config.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Copy config file to experiment directory
    dest_config = exp_dir / "config.json"
    shutil.copy(config_path, dest_config)
    print(f"Copied config to {dest_config}")

    return exp_dir


def train_model(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: GPTConfig,
    exp_dir: Path,
):
    # Get device and move model
    device = get_device()
    print(f"Using device: {device}")
    print(f"Mask hints: {config.mask_hints}")

    # Check if bf16 is supported
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if use_bf16:
        print("Using bf16 precision")
    else:
        print("Using fp32 precision")

    model = model.to(device)

    # Convert model to bf16 if supported
    if use_bf16:
        model = model.to(torch.bfloat16)

    total_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Parameters: {total_params:,}")

    # Setup metrics path in experiment directory
    metrics_path = exp_dir / "metrics.csv"
    print(f"Saving metrics to {metrics_path}")

    pbar = trange(config.max_iters)

    train_loss = Metric("train_loss", average="exponential", weight=0.99)
    train_acc = Metric("train_acc", average="exponential", weight=0.99)
    train_acc_full = Metric("train_acc_full", average="exponential", weight=0.99)

    train_iter = iter(train_loader)

    best_valid_loss = float("inf")
    best_model_path = exp_dir / "best_model.pt"

    for iter_num in pbar:
        try:
            x, y_true = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y_true = next(train_iter)

        # Move data to device
        x = x.to(device)
        y_true = y_true.to(device)

        model.zero_grad()

        # Forward pass: (B, 81) -> (B, 81, 9)
        y_hat = model(x)

        # Compute loss with optional hint masking
        loss = compute_loss(criterion, y_hat, y_true, x, mask_hints=config.mask_hints)
        loss.backward()
        optimizer.step()

        # Compute metrics
        with torch.no_grad():
            train_loss.update(loss.item())
            train_acc.update(compute_accuracy(y_hat, y_true, x))
            train_acc_full.update(compute_full_accuracy(y_hat, y_true, x))

        pbar.set_postfix_str(
            f"loss: {train_loss.corrected_value:.4f} "
            f"acc: {train_acc.corrected_value:.2%} "
            f"acc_full: {train_acc_full.corrected_value:.2%}"
        )

        if iter_num % config.eval_interval == 0:
            valid_loss, valid_acc, valid_acc_full = eval_model(
                model, criterion, valid_loader, device, config.mask_hints, use_bf16
            )

            # Save best model
            if valid_loss.corrected_value < best_valid_loss:
                best_valid_loss = valid_loss.corrected_value
                torch.save(
                    {
                        "iter": iter_num,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_valid_loss,
                    },
                    best_model_path,
                )

            all_metrics = [
                train_loss,
                train_acc,
                train_acc_full,
                valid_loss,
                valid_acc,
                valid_acc_full,
            ]
            save_metrics(all_metrics, metrics_path, iter_num)
            for metric in all_metrics:
                metric.clear()


def save_metrics(metrics: list[Metric], metrics_path: Path, iter_num: int):
    vals = OrderedDict()
    vals["iter"] = iter_num
    vals["time"] = time()
    for metric in metrics:
        vals[metric.name] = metric.corrected_value

    file_exists = metrics_path.exists()

    with open(metrics_path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=vals.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(vals)


def eval_model(
    model, criterion, valid_loader, device, mask_hints=False, use_bf16=False
):
    valid_loss = Metric("valid_loss", average="arithmetic")
    valid_acc = Metric("valid_acc", average="arithmetic")
    valid_acc_full = Metric("valid_acc_full", average="arithmetic")
    model.eval()

    with torch.no_grad():
        for x, y_true in valid_loader:
            # Move data to device
            x = x.to(device)
            y_true = y_true.to(device)

            # Forward pass
            y_hat = model(x)

            # Compute loss with optional hint masking
            loss = compute_loss(criterion, y_hat, y_true, x, mask_hints=mask_hints)

            valid_loss.update(loss.item())
            valid_acc.update(compute_accuracy(y_hat, y_true, x))
            valid_acc_full.update(compute_full_accuracy(y_hat, y_true, x))

    model.train()
    return valid_loss, valid_acc, valid_acc_full
