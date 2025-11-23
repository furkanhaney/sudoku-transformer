import json
import argparse
import torch
from pathlib import Path
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from gpt import GPT
from data import create_dataloaders
from training import train_model, setup_experiment_dir
from utils import GPTConfig


def main():
    parser = argparse.ArgumentParser(description="Train Sudoku solver")
    parser.add_argument(
        "--config",
        type=str,
        default="config/gpt2_sm.json",
        help="Path to config JSON file",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = GPTConfig(**config_dict)
    print(f"Loaded config from {config_path}")
    print(f"Experiment: {config.experiment_name}")

    # Setup experiment directory
    exp_dir = setup_experiment_dir(config, config_path)

    # Create model
    model = GPT(
        vocab_size=10,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_positions=81,
        dropout=config.dropout,
        output_dim=9,
    )

    # Compile model for faster training (PyTorch 2.0+)
    if config.use_compile:
        print(f"Compiling model with mode: {config.compile_mode}")
        model = torch.compile(model, mode=config.compile_mode)
    else:
        print("Running without torch.compile()")

    # Create dataloaders
    train_loader, valid_loader = create_dataloaders(
        batch_size=config.batch_size, num_workers=config.num_workers
    )

    # Setup training
    criterion = CrossEntropyLoss(reduction="none")
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # Train
    train_model(
        model, criterion, optimizer, train_loader, valid_loader, config, exp_dir
    )


if __name__ == "__main__":
    main()
