# Unique-puzzle RTX 5090 campaign — 2026-09-20

This directory records the first completed, predeclared Axis run on the
uniqueness-enforcing generated Sudoku stream. The run used public
`sudoku-transformer` commit
`0fd23e7938d853df048598136de638d9a8f6b643` and exact Axis commit
`d53d3354b989e04aabd68d032bd841111c88ad03`.

## Completed audited run

```bash
bash scripts/train.sh \
  --steps 500 --batch 256 \
  --eval-size 512 --eval-batch 256 --eval-every 100 \
  --audit-size 1024 --log-every 25 \
  --embedding 12 --heads 3 --layers 1 --blanks 36 \
  --learning-rate 3e-3 --weight-decay 1e-2
```

The 3,129-parameter FP32 model consumed 128,000 unique training boards. The
command began at `2026-09-20T15:43:32Z` and ended at
`2026-09-20T16:03:49Z`: `1,216.94 s` (`20m 16.94s`) end to end, below the
declared 30-minute limit. The trainer reported `1,193.46 s` through the final
tuning evaluation.

| population | boards | loss | blank accuracy | exact puzzles |
|---|---:|---:|---:|---:|
| tuning, initial | 512 | 2.813295 | 11.73% | 0.00% |
| tuning, final | 512 | 2.137585 | 17.19% | 0.00% |
| untouched audit | 1,024 | 2.137709 | 17.36% | 0.00% |

The audit used its third seed namespace and was first evaluated after training.
Receipts report zero tuning/audit overlap, zero overlap between all 1,536
evaluation IDs and the 128,000 training IDs, and `0.0000%` training-ID reuse.
This is a measured learning result on uniquely solvable generated puzzles. It
did not reach the historical accuracy target and is not a solving result.

One-second telemetry contains 1,181 samples: mean/peak GPU utilization
`3.38%` / `64%`, peak memory `1,378 MiB`, mean/peak power `76.09 W` /
`82.64 W`, and peak temperature `39 C`. The full command and receipts are in
`final-tiny-b256-s500.log`; `gpu-telemetry.csv` is the raw telemetry.

## Performance findings

The uniqueness generator was not the bottleneck. Disposable timing
instrumentation measured roughly `3.3 ms` to generate 256 unique boards and
about `0.5 ms` to construct/upload their tensors, while the then-current
206,537-parameter trainer step took about 20 seconds.

The campaign drove a sequence of exact Axis revisions. For the same
embedding-64, two-head, four-layer, batch-256 shape:

| Axis revision | measured steady trainer step | 400-step projection |
|---|---:|---:|
| `e78783a` 64x64 GEMM tiles | 18.96 s | 126.4 min |
| `9ba184d` BF16 matrix inputs | 19.00 s | 126.7 min |
| `d53d335` one trainer-boundary sync | 20.70 s | 138.0 min |
| `7f61356` compiled-plan cache | 6.47 s | 43.2 min |

The cached-plan revision is a 3.2x steady-step improvement over `d53d335`, but
the larger model still did not meet the 30-minute end-to-end constraint.
Batch 512 was legal on `9ba184d` but provided no throughput gain; batches 1,024
and 2,048 failed with the preserved 16,777,216-contribution planner limit.

`strace` could not collect CPU samples because the provider kernel set
`perf_event_paranoid=4`. Its syscall and execution receipts still identified
57 `tileiras` plus 57 `ptxas` compilations during a one-step process. Compilation
occurred once per process; steady steps remained limited by launch and
synchronization behavior. The raw benchmark, batch-probe, and profiling logs
are retained beside the final run.

## Stopped exploratory arm

An embedding-64, two-head, four-layer batch-256 arm on Axis `dbe1c0d` was
stopped at step 60 when the end-to-end 30-minute requirement was introduced.
It consumed 15,360 unique boards in `1,241.89 s`, but never reached a fixed
evaluation or audit checkpoint. Its curve is not used for a model-quality
claim; the log and telemetry remain as performance evidence.

## Rental

- pod: `fi1e4cl5kzyreo` (`axis-long-5090-20260920-140356`)
- hardware: NVIDIA GeForce RTX 5090, 32,607 MiB, driver 580.65.06
- image: Ubuntu 24.04.3 LTS; Rust 1.98.1; repository-local CUDA 13.2
- provider rate: `$0.69/hour`, below the authorized `$1.00/hour` ceiling
- provider deadline: `2026-09-20T18:03:56Z`
- allocated: `2026-09-20T14:03:56Z`
- deletion absence verified: `2026-09-20T16:08:53.664849Z`
- allocation-to-absence interval: `7,497.665 s` (`2.082685 h`)
- estimated charge at the quoted rate: `$1.4371`

`terminated.json` records `absent: true`, and `foreground-termination.log`
records the guard's successful deletion and absence check. The subsequent
account pod list does not contain this campaign's pod ID. It contains one
unrelated running pod named `openwebui-qwen38-gemma4-20260920-072107`, so the
receipt proves exact-pod absence rather than an account-wide zero count; that
other rental was not touched.
