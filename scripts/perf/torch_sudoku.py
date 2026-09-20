#!/usr/bin/env python3
"""Steady-step PyTorch reference for the Axis Sudoku Transformer shapes."""
import argparse
import json
import statistics
import time

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, x):
        batch, positions, width = x.shape
        shape = (batch, positions, self.heads, self.head_width)
        q = self.query(x).view(shape).transpose(1, 2)
        k = self.key(x).view(shape).transpose(1, 2)
        v = self.value(x).view(shape).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / self.head_width**0.5
        attended = scores.softmax(-1) @ v
        return self.output(attended.transpose(1, 2).reshape(batch, positions, width))


class Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * width, width),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class SudokuTransformer(nn.Module):
    def __init__(self, width: int, heads: int, layers: int):
        super().__init__()
        self.token = nn.Linear(10, width)
        self.position = nn.Parameter(torch.empty(81, width).uniform_(-0.02, 0.02))
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 9)

    def forward(self, x):
        x = self.token(x) + self.position
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--embedding", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is required")

    torch.manual_seed(42)
    device = torch.device("cuda")
    model = SudokuTransformer(args.embedding, args.heads, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    tokens = torch.randint(0, 10, (args.batch, 81), device=device)
    inputs = torch.nn.functional.one_hot(tokens, 10).float()
    targets = torch.randint(0, 9, (args.batch, 81), device=device)
    mask = torch.rand((args.batch, 81), device=device) < (36 / 81)

    def step():
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16)
        with context:
            logits = model(inputs)
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 9), targets.reshape(-1), reduction="none"
            ).reshape(args.batch, 81)
            loss = losses[mask].mean()
        loss.backward()
        optimizer.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    durations = []
    for _ in range(args.steps):
        started = time.perf_counter()
        step()
        torch.cuda.synchronize()
        durations.append((time.perf_counter() - started) * 1_000)
    print(json.dumps({
        "backend": "torch",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "bf16" if args.bf16 else "fp32",
        "batch": args.batch,
        "embedding": args.embedding,
        "heads": args.heads,
        "layers": args.layers,
        "median_step_ms": statistics.median(durations),
        "min_step_ms": min(durations),
        "steps": len(durations),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
