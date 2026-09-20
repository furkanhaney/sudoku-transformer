# GENERATED — edit HUB.md, then: scripts/hubgen.py --write
- Depth lives in docs/ spokes: read the spoke before CHANGING what it covers.
- Launch from a node, never inside a src/. Only src/ recurses.
- Slots exist only when nonempty. Ephemera (build/, scratch/, worktrees) are
  gitignored; archive/ is frozen.
- Put a thing at the LCA of its consumers. Everything enters at the bottom.
- Filesystem should resemble the shape of the project, Conway style:
  give real crates, programs, and subsystems their own folders.
- Internal structure changes are clean migrations: update callers and
  remove retired paths; do not retain compatibility shims.
- DER00: Leave the code you touched a tiny bit better. The requested change
  can supply the improvement; no extra cleanup is owed. Preserve required
  behavior except intended changes. Do not expand scope, invent abstractions,
  or move complexity elsewhere merely to satisfy this rule.
- Read and improve the owning scope's docs/ as you learn. Roughly every 5-25
  turns, fold useful findings, ideas, corrections, or next steps into local
  docs; a small edit is enough. Do not rely on chat or compaction alone.
- Child hubs only ADD — never restate or override an ancestor.
- Ratchet baselines only go down; an exception needs a written waiver.
- Verify by driving the real thing; report red gates verbatim.
- Commit frequently: small working increments, in the child repo that owns
  the change. Never end a task with the work only on disk.
- Shell cwd persists across tool calls and resets between them unpredictably:
  use absolute paths; never trust a bare `cd`.
- Doctrine: <root>/docs/ · why hubs look like this: <root>/docs/hubs.md
# --- end invariant — this node's half follows ---
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

Axis is an outside crates.io dependency pinned by Cargo.lock. Shared tensor,
optimizer, data-regime, or module behavior belongs in Axis; Sudoku-specific
architecture, generation, metrics, plots, and experimental choices belong here.
