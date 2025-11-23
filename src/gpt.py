import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention without causal masking (bidirectional)."""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # Query, key, value projections for all heads in batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Flash Attention: uses optimized CUDA kernels when available
        # Automatically falls back to standard implementation if not available
        # This is mathematically equivalent but much faster and more memory efficient
        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ with Flash Attention support
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,  # No causal mask for bidirectional attention
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Fallback to manual implementation for older PyTorch versions
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = torch.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, hs)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-layer normalization (GPT-2 style)."""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, dropout)

    def forward(self, x):
        # Pre-norm: normalize before attention/mlp
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    GPT-2 style transformer without causal masking for Sudoku solving.

    Input: (batch_size, 81) integers 0-9 representing Sudoku puzzle
    Process:
        1. Embed tokens (0-9) to n_embd dimensions
        2. Add positional embeddings (81 positions)
        3. Pass through transformer blocks (bidirectional attention)
        4. Project to output classes for each position
    Output: (batch_size, 81, 9) logits for digits 1-9 (classes 0-8)
    """

    def __init__(
        self,
        vocab_size: int = 10,
        n_embd: int = 512,
        n_layer: int = 8,
        n_head: int = 8,
        n_positions: int = 81,
        dropout: float = 0.1,
        output_dim: int = 9,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_positions = n_positions

        # Token embeddings: map tokens 0-9 to embedding dimension
        self.token_emb = nn.Embedding(vocab_size, n_embd)

        # Positional embeddings (learnable)
        self.pos_emb = nn.Embedding(n_positions, n_embd)

        # Dropout
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(n_embd)

        # Output projection: (B, 81, n_embd) -> (B, 81, 9)
        self.output_proj = nn.Linear(n_embd, output_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        B, T = x.size()  # batch size, sequence length (81)

        # Token embeddings: (B, 81) -> (B, 81, n_embd)
        tok_emb = self.token_emb(x)

        # Positional embeddings: (81,) -> (81, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.pos_emb(pos)

        # Combine embeddings and apply dropout
        x = self.drop(tok_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Output projection: (B, 81, n_embd) -> (B, 81, 9)
        x = self.output_proj(x)

        return x
