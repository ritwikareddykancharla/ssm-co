# --------------------------------------------------
# Minimal TSP model + rollout (device-agnostic)
# - GPU-optimized (no TPU/XLA assumptions)
# - Safer attention + faster logits
# --------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ==================================================
# Sparse kNN Attention (STATIC ENCODER)
# ==================================================
class SparseKNNGraphAttention(nn.Module):
    def __init__(self, dim, k=8):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.k = k

    def forward(self, x):
        B, N, D = x.shape

        qkv = self.qkv(x).view(B, N, 3, D)
        q, k, v = qkv.unbind(dim=2)

        # cosine similarity for kNN graph
        sim = torch.matmul(
            F.normalize(k, dim=-1),
            F.normalize(k, dim=-1).transpose(-1, -2)
        )

        sim = sim - 1e9 * torch.eye(N, device=x.device)

        k_val = min(self.k, N - 1)
        _, idx = sim.topk(k_val, dim=-1)  # [B, N, k]

        k_exp = k.unsqueeze(2).expand(B, N, N, D)
        v_exp = v.unsqueeze(2).expand(B, N, N, D)
        idx_exp = idx.unsqueeze(-1).expand(B, N, k_val, D)

        k_sel = k_exp.gather(2, idx_exp)
        v_sel = v_exp.gather(2, idx_exp)

        scores = (q.unsqueeze(2) * k_sel).sum(-1) / math.sqrt(D)
        attn = torch.softmax(scores, dim=-1)

        out = (attn.unsqueeze(-1) * v_sel).sum(2)
        return x + self.out(out)


# ==================================================
# Static Graph Encoder
# ==================================================
class GraphEncoder(nn.Module):
    def __init__(self, dim=128, layers=2):
        super().__init__()
        self.input = nn.Linear(2, dim)
        self.layers = nn.ModuleList(
            [SparseKNNGraphAttention(dim) for _ in range(layers)]
        )

    def forward(self, coords):
        h = self.input(coords)
        for layer in self.layers:
            h = layer(h)
        return h


# ==================================================
# Token-level SSM Decoder (Mamba-style)
# ==================================================
class SSMBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )
        self.A = nn.Parameter(torch.randn(dim) * 0.01)
        self.B = nn.Parameter(torch.randn(dim) * 0.01)

    def forward(self, x, state):
        h = self.norm(x)
        y = self.ff(h)

        if state is None:
            state = torch.tanh(self.B * y)
        else:
            state = torch.tanh(self.A * state + self.B * y)

        return x + y, state


# ==================================================
# Full TSP Model (policy-only, no value head)
# ==================================================
class TSPModel(nn.Module):
    def __init__(self, dim=256, layers=4):
        super().__init__()
        self.encoder = GraphEncoder(dim)
        self.decoder = nn.ModuleList([SSMBlock(dim) for _ in range(layers)])
        self.query = nn.Linear(dim, dim)

    def rollout(self, coords, greedy=False):
        """
        coords: (B, N, 2)
        returns: log_probs, entropies, tour_length
        """
        B, N, _ = coords.shape

        node_emb = self.encoder(coords).contiguous()  # (B, N, D)

        visited = torch.zeros(B, N, dtype=torch.bool, device=coords.device)
        visited[:, 0] = True

        current = torch.zeros(B, dtype=torch.long, device=coords.device)
        token = node_emb[:, 0:1, :]
        states = [None] * len(self.decoder)

        log_probs = torch.zeros(B, device=coords.device)
        entropies = torch.zeros(B, device=coords.device)
        tour_len = torch.zeros(B, device=coords.device)

        for _ in range(N - 1):
            h = token
            for i, layer in enumerate(self.decoder):
                h, states[i] = layer(h, states[i])

            # faster than einsum on GPU
            q = self.query(h).squeeze(1)               # (B, D)
            logits = torch.bmm(node_emb, q.unsqueeze(-1)).squeeze(-1)  # (B, N)

            logits = logits.masked_fill(visited, torch.finfo(logits.dtype).min)
            # logits = logits.masked_fill(visited, -1e9)
            dist = Categorical(logits=logits)

            nxt = logits.argmax(dim=-1) if greedy else dist.sample()

            if not greedy:
                log_probs += dist.log_prob(nxt)
                entropies += dist.entropy()

            prev = current
            tour_len += torch.norm(
                coords[torch.arange(B), prev]
                - coords[torch.arange(B), nxt],
                dim=-1      
            )

            visited = visited.clone()
            visited[torch.arange(B), nxt] = True
            current = nxt
            token = node_emb[torch.arange(B), nxt].unsqueeze(1)

        tour_len += torch.norm(
            coords[torch.arange(B), current] - coords[:, 0], dim=-1
        )

        return log_probs, entropies, tour_len
