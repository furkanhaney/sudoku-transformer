# Sudoku Transformer — generated-data research on Axis

This is the Rust + Axis successor to the 2025 Python + PyTorch experiment.
README.md is for outside humans; keep it visual, direct, and limited to measured
claims. Agent contracts, migration details, and continuation state live here or
under docs/.

The scientific object is a bidirectional transformer that fills Sudoku blanks.
Training data is generated indefinitely from valid boards. Assert draw-ID IDR
before delivery, keep canonical puzzle identities disjoint across training,
tuning, and audit populations, and report blank-cell accuracy separately from
exact whole-board solves.

Run `bash scripts/check.sh`. A smoke must pass before a longer experiment. Put
small durable logs in `models/<experiment>/`; raw datasets and large checkpoints
stay ignored. Every public number needs its command, hardware, and log.

Axis is an outside git dependency pinned by revision. Shared tensor, optimizer,
data-regime, or module behavior belongs in Axis; Sudoku-specific architecture,
generation, metrics, plots, and experimental choices belong here.
