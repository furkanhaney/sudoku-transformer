# Migration contract

Source: <https://github.com/furkanhaney/sudoku-transformer>.

Preserved in the acceptance path:

- algorithmically valid 9x9 Sudoku boards and digit relabeling;
- fresh randomized clue masks;
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
GELU, f32 throughout, a generated stream in place of the fixed Kaggle corpus,
and no dropout or compilation. The generator supplies an IDR acceptance that
the finite source cannot: every delivered draw has a fresh stable identity,
train/evaluation seeds occupy disjoint identity namespaces, and both guards
emit receipts. This proves the declared operational regime, not independence
or infinite informational diversity.
