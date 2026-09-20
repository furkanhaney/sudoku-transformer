#!/usr/bin/env python3
"""Render the measured README board and current learning curve."""

from pathlib import Path
import re

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INK = "#172033"
ACCENT = "#5b5bd6"
ORANGE = "#ee8a18"
PAPER = "#ffffff"
GRID = "#d9ddea"

BOARD = [
    [0, 0, 3, 4, 0, 6, 7, 0, 9],
    [0, 5, 6, 0, 8, 0, 1, 2, 0],
    [7, 0, 0, 1, 0, 3, 4, 0, 6],
    [0, 3, 4, 0, 0, 7, 8, 0, 1],
    [0, 6, 0, 8, 0, 1, 2, 0, 4],
    [0, 9, 0, 2, 0, 4, 5, 0, 7],
    [0, 4, 5, 0, 7, 0, 9, 0, 2],
    [0, 7, 0, 9, 1, 0, 3, 0, 5],
    [0, 1, 2, 0, 4, 0, 6, 7, 0],
]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
        }
    )


def render_board() -> None:
    figure, axis = plt.subplots(figsize=(8, 8), facecolor=PAPER)
    axis.set_facecolor(PAPER)
    for index in range(10):
        width = 3.2 if index in (0, 9) else 2.8 if index % 3 == 0 else 0.8
        color = ORANGE if index in (0, 9) else ACCENT if index % 3 == 0 else INK
        axis.plot([index, index], [0, 9], color=color, linewidth=width)
        axis.plot([0, 9], [index, index], color=color, linewidth=width)
    for row, values in enumerate(BOARD):
        for column, value in enumerate(values):
            if value:
                axis.text(column + 0.5, 8.5 - row, str(value), ha="center", va="center", fontsize=24)
    axis.set(xlim=(-0.05, 9.05), ylim=(-0.05, 9.65), aspect="equal")
    axis.axis("off")
    axis.set_title("A fresh board from the Axis acceptance stream", loc="left", pad=14, fontsize=16)
    figure.tight_layout(pad=0.8)
    figure.savefig(ROOT / "img/axis_generated_puzzle.png", dpi=150, facecolor=PAPER)
    plt.close(figure)


def render_curve() -> None:
    log = (ROOT / "models/runpod-5090-unique-20260920/final-tiny-b256-s500.log").read_text()
    train = [(int(step), float(loss)) for step, loss in re.findall(r"step=(\d+).*pre_update_loss=([0-9.]+)", log)]
    initial = float(re.search(r"initial eval_loss=([0-9.]+)", log).group(1))
    initial_accuracy = float(re.search(r"initial eval_loss=[0-9.]+ blank_accuracy=([0-9.]+)%", log).group(1))
    final = float(re.search(r"final eval_loss=([0-9.]+)", log).group(1))
    checkpoints = [
        (int(step), float(loss), float(accuracy))
        for step, loss, accuracy in re.findall(
            r"evaluation step=(\d+) eval_loss=([0-9.]+) blank_accuracy=([0-9.]+)%",
            log,
        )
    ]
    final_accuracy = float(re.search(r"final eval_loss=[0-9.]+ blank_accuracy=([0-9.]+)%", log).group(1))
    evaluation = [(0, initial, initial_accuracy), *checkpoints, (train[-1][0], final, final_accuracy)]

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=PAPER)
    for axis in (loss_axis, accuracy_axis):
        axis.set_facecolor(PAPER)
        axis.grid(color=GRID, linewidth=0.8)
        for spine in axis.spines.values():
            spine.set_color(INK)
            spine.set_alpha(0.45)

    loss_axis.plot([step for step, _ in train], [loss for _, loss in train], color=ACCENT, linewidth=2.2, alpha=0.8, label="fresh-batch training")
    loss_axis.plot([step for step, _, _ in evaluation], [loss for _, loss, _ in evaluation], color=ORANGE, linewidth=2.5, marker="o", markersize=6, label="fixed tuning set")
    loss_axis.axhline(2.197225, color=INK, alpha=0.55, linestyle=":", linewidth=2, label="uniform 9-class loss")
    loss_axis.set_xlabel("optimizer step (256 fresh boards each)")
    loss_axis.set_ylabel("blank-only cross-entropy")
    loss_axis.legend(frameon=False, fontsize=9)

    accuracy_axis.plot([step for step, _, _ in evaluation], [accuracy for _, _, accuracy in evaluation], color=ACCENT, linewidth=2.5, marker="o", markersize=6)
    accuracy_axis.set_xlabel("optimizer step")
    accuracy_axis.set_ylabel("fixed tuning blank accuracy (%)")
    accuracy_axis.set_ylim(0, 20)
    accuracy_axis.text(0.98, 0.08, "exact puzzles: 0 / 512", transform=accuracy_axis.transAxes, ha="right", color=INK)

    figure.suptitle("Axis Sudoku — 128,000 unique training boards on RTX 5090", x=0.06, ha="left", fontsize=16, fontweight="semibold")
    figure.text(0.06, 0.91, "Uniquely solvable generated puzzles · fixed 512-board tuning population", color=INK)
    for spine in loss_axis.spines.values():
        spine.set_color(INK)
        spine.set_alpha(0.45)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(ROOT / "img/axis_acceptance_25.png", dpi=150, facecolor=PAPER)
    plt.close(figure)


if __name__ == "__main__":
    style()
    render_board()
    render_curve()
