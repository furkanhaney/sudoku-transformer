# Performance contract

Performance is an acceptance property of this migration, alongside numerical
correctness and the IDR receipts. The reference is the equivalent PyTorch CUDA
training step at the same batch size, embedding width, head count, layer count,
dtype, optimizer, and masked loss. Both paths synchronize CUDA around timed
steps. Setup and two warmup steps are excluded.

Run the paired benchmark on a CUDA host with PyTorch installed:

```bash
bash scripts/perf/compare_torch.sh
```

Override `BATCH`, `EMBEDDING`, `HEADS`, `LAYERS`, and `STEPS` to measure another
shape. The script emits machine-readable JSON for each backend. A benchmark is
not a fixed pass/fail gate yet: preserve receipts, compare the same hardware,
and ratchet steady-step time downward without weakening the model or workload.

The September 20 RTX 5090 profile for the 206k-parameter, FP32, batch-256 model
measured 20.70 seconds per steady step before repeated-plan caching and 6.473
seconds afterward, a 3.20x improvement. That still projects to 43.15 minutes
for 400 training steps before evaluation, so the requested sub-30-minute target
remains open. Profiling should next separate graph construction, kernel launch,
backward contraction, optimizer updates, and synchronization; wall-clock step
time alone cannot identify the next dominant operation.
