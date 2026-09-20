# Experiment ladder

Every rung uses fresh generated boards, a fixed evaluation stream, 36 blanks,
AdamW, and blank-only loss. Advance only after the smaller rung completes with
finite loss and passing IDR/disjointness receipts.

| rung | embedding | heads | layers | batch | purpose |
|---|---:|---:|---:|---:|---|
| smoke | 12 | 3 | 1 | 2 | complete API and CUDA acceptance |
| local | 24 | 4 | 2 | 4 | learning-curve iteration |
| 5090-small | 64 | 8 | 4 | 8 | first rented-GPU scaling run |
| 5090-medium | 128 | 8 | 6 | 8 | depth/width comparison after small learns |

The current correctness-first Axis contraction uses indexed gather/reduce plans
with a 16,777,216-contribution bound. For self-attention the dominant plan grows
roughly as `batch * positions^2 * embedding`, so original PyTorch batch sizes do
not yet fit. Record the first rejected configuration as a compiler/backend
finding rather than silently shrinking it.

Evaluation is generated once from its disjoint namespace and can be much larger
than one backend contraction plan. `--eval-batch` chunks model execution while
preserving one aggregate metric over the declared `--eval-size`; `--eval-every`
measures that same fixed population during a long run.

Suggested rented-GPU command:

```bash
bash scripts/train.sh \
  --steps 5000 --batch 8 --eval-size 512 --eval-batch 8 --eval-every 250 \
  --embedding 64 --heads 8 --layers 4 \
  --blanks 36 --learning-rate 1e-3 --weight-decay 1e-2
```

A run earns a public learning claim only when fixed held-out loss improves and
its log records hardware, exact commit, complete command, elapsed time, sample
count, IDR receipt, and train/evaluation overlap receipt.
