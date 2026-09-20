# RunPod RTX 5090 run — 2026-09-20

This directory records a bounded Axis and Sudoku experiment on public
`sudoku-transformer` commit `35cfb9eb42c59c42debe7e8a3de0084ae21235d3`
and its exact Axis pin
`767326049ef661e226a074f3fe551ebcd98613bf`.

## Rental receipt

- RunPod pod: `3xnqhxp38a13u5` (`axis-sudoku-5090-20260920-133800`)
- provider rate: `$0.69/hour`, below the authorized `$1.00/hour` ceiling
- created: `2026-09-20T13:38:01.189Z`
- deletion absence verified: `2026-09-20T13:53:08.069Z`
- allocation-to-absence interval: `906.880 s` (`15m 6.880s`)
- estimated GPU charge at the provider rate: `$0.1738`
- teardown: `terminated.json` records the deletion receipt and
  `active-pods-after-termination.json` records zero active pods

The independent deadline guard ran as the user-systemd unit
`axis-runpod-guard-20260920.service`; the pod also carried the provider-side
`2026-09-20T15:08:00Z` termination deadline. The local guard was stopped only
after foreground deletion and absence verification succeeded.

## Hardware and toolchain

- NVIDIA GeForce RTX 5090 (Blackwell), 32,607 MiB VRAM
- NVIDIA driver `580.65.06`
- AMD Ryzen Threadripper PRO 7955WX, 16 cores / 32 threads
- 188 GiB host RAM
- Ubuntu 24.04.3 LTS
- Rust `1.98.1`
- repository-local CUDA `13.2.51` and cuTile `0.3.1`

The pinned RunPod image contained CUDA 12.8 runtime support but did not put
`nvcc` on `PATH`. `scripts/setup_cuda.py` installed and hash-verified Axis's
minimal CUDA 13.2 toolkit. The first locked framework build then failed with:

```text
Unable to find libclang: "couldn't find any valid shared libraries matching: ['libclang.so', 'libclang-*.so', 'libclang.so.*', 'libclang-*.so.*']"
```

Installing Ubuntu's `libclang-dev` was sufficient. The unmodified retry passed
12 host-side tests, and the explicit CUDA run passed all 8 ignored GPU oracle
tests. `framework-test.log` preserves the prerequisite failure;
`framework-test-rerun.log` and `framework-cuda-tests.log` preserve the passing
runs.

## Axis framework training witness

Command:

```bash
bash src/addition/scripts/train.sh --steps 500
```

The 500-step generated-stream run consumed 128,000 unique training draws.
Held-out MSE fell from `0.80632222` to `0.00387844` in `1.29 s` of reported
training time. The IDR receipt reported `0.0000%` reuse, and the disjointness
receipt reported zero overlap with 256 evaluation IDs. See
`addition-500.log`.

## Scaled tiny Sudoku run

Command (wrapped in a 2,400-second process timeout):

```bash
bash scripts/train.sh \
  --steps 200 \
  --batch 8 \
  --eval-size 8 \
  --embedding 12 \
  --heads 3 \
  --layers 1 \
  --blanks 36 \
  --learning-rate 3e-3 \
  --weight-decay 1e-2
```

This is 8 times the local 25-step run and 4 times its batch size. It consumed
1,600 fresh boards in `51.27 s`. Held-out blank-only loss fell from `2.873419`
to `2.170745`, blank-cell accuracy moved from `12.50%` to `14.93%`, and exact
solve rate remained `0.00%`. IDR reported all 1,600 IDs unique with zero reuse;
train/evaluation disjointness reported zero overlap with 8 evaluation IDs.
The raw record is `sudoku-200-b8-e8.log`.

One-second `nvidia-smi` samples in `gpu-telemetry.csv` recorded 60 samples:
mean/peak GPU utilization `2.25%` / `6%`, peak allocated memory `866 MiB`,
mean/peak power `57.21 W` / `63.60 W`, and peak temperature `28 C`. For this
small model the current synchronous lowering is launch/synchronization bound;
the receipt does not demonstrate RTX 5090 throughput saturation.

## Larger Sudoku model

Command (wrapped in a 900-second process timeout):

```bash
bash scripts/train.sh \
  --steps 100 \
  --batch 4 \
  --eval-size 4 \
  --embedding 24 \
  --heads 4 \
  --layers 2 \
  --blanks 36 \
  --learning-rate 1e-3 \
  --weight-decay 1e-2
```

The 16,929-parameter model consumed 400 unique boards in `66.05 s`. Held-out
loss fell from `3.056971` to `2.197362`, while blank accuracy moved from
`11.81%` to `11.11%` and exact solves remained `0.00%`. The receipts again
reported zero reuse and zero train/evaluation overlap. This is a complete
mechanics witness, not Sudoku-solving evidence. See
`sudoku-100-b4-24x2.log` and `gpu-telemetry-large.csv`.

The larger run's 74 one-second samples recorded mean/peak GPU utilization
`1.91%` / `8%`, peak allocated memory `866 MiB`, mean/peak power `58.05 W` /
`61.60 W`, and peak temperature `30 C`.

## Measured backend limit

Before the successful scaled run, evaluation sizes 64 and 16 were attempted
with training batch 8. Both stopped before the initial metric with the same
error:

```text
Error: "index plan must have 1..=16777216 contributions (experimental backend limit)"
```

The exact failed commands are preserved in `sudoku-200-b8.log` and
`sudoku-200-b8-e16.log`. Matching evaluation size to batch size 8 completed.
This receipt establishes the failure trigger for these shapes; it does not yet
identify whether the intended contract should support a separately sized
evaluation batch or reject it earlier with a shape-specific message.

`hardware-and-revisions.log` contains the direct hardware and revision probe.
`pod.json`, `terminated.json`, and `active-pods-after-termination.json` contain
only provider-safe identifiers, rate/deadline metadata, and teardown state.
