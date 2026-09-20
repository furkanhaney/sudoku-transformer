# Migration contract

Source: <https://github.com/furkanhaney/sudoku-transformer>.

Preserved in the acceptance path:

- algorithmically valid 9x9 Sudoku boards and digit relabeling;
- fresh randomized clue masks accepted only after exact uniqueness counting;
- token plus learned position embeddings;
- non-causal multi-head self-attention;
- pre-LayerNorm residual attention and feed-forward paths;
- four-times-width GELU feed-forward expansion;
- nine-class logits at every position;
- cross-entropy restricted to input zeros;
- accuracy restricted to blanks and exact whole-puzzle solve rate;
- AdamW optimization; and
- separate training and evaluation populations.

The bounded Axis configuration is deliberately smaller. It uses tanh-approximated
GELU, a generated stream in place of the fixed Kaggle corpus, and no dropout.
The default path is FP32 throughout; `--bf16` rounds matrix-product inputs to
BF16 while retaining FP32 accumulation and FP32 optimizer state. The generator
supplies an IDR acceptance that the finite source cannot: every delivered draw
has a fresh stable identity. The Sudoku program canonicalizes each clue board
by first-occurrence digit relabeling, then an Axis disjointness ledger compares
those supplied problem identities across training, tuning, and final-audit
populations. It can therefore catch one logical puzzle arriving from distinct
generator seeds. Row and column positions remain meaningful in version 1 of
this app-owned identity. The receipts prove the declared observed regime, not
unseen separation, independence, or infinite informational diversity.

Uniqueness is part of the target contract. Without it, exact agreement with the
generator's hidden completion can be impossible even for a correct Sudoku
solver. The generator counts solutions up to two and delivers only puzzles with
exactly one.
