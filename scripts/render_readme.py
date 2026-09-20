#!/usr/bin/env python3
"""Render the two measured README figures with one visual language."""

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
    log = (ROOT / "models/axis_tiny_00/metrics.log").read_text()
    train = [(int(step), float(loss)) for step, loss in re.findall(r"step=(\d+).*pre_update_loss=([0-9.]+)", log)]
    initial = float(re.search(r"initial eval_loss=([0-9.]+)", log).group(1))
    final = float(re.search(r"final eval_loss=([0-9.]+)", log).group(1))

    figure, axis = plt.subplots(figsize=(11, 5.5), facecolor=PAPER)
    axis.set_facecolor(PAPER)
    axis.plot([step for step, _ in train], [loss for _, loss in train], color=ACCENT, linewidth=2.5, label="fresh-batch training loss")
    axis.plot([0, train[-1][0]], [initial, final], color=ORANGE, linestyle="--", linewidth=2, marker="o", markersize=7, label="fixed evaluation loss")
    axis.axhline(2.197225, color=INK, alpha=0.55, linestyle=":", linewidth=2, label="uniform 9-class loss")
    axis.grid(color=GRID, linewidth=0.8)
    for spine in axis.spines.values():
        spine.set_color(INK)
        spine.set_alpha(0.45)
    axis.set_title("Axis Sudoku acceptance — first 50 generated boards", loc="left", fontsize=16)
    axis.set_xlabel("optimizer step (2 fresh boards each)")
    axis.set_ylabel("blank-only cross-entropy")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(ROOT / "img/axis_acceptance_25.png", dpi=150, facecolor=PAPER)
    plt.close(figure)


if __name__ == "__main__":
    style()
    render_board()
    render_curve()
