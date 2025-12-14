# --------------------------------------------------
# Minimal TSP toy experiment (TPU / XLA optimized)
# Single-file, runnable, clean, AUTOGRAD-SAFE
# --------------------------------------------------

import math
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import torch_xla
import torch_xla.core.xla_model as xm

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

        sim = torch.matmul(
            F.normalize(k, dim=-1),
            F.normalize(k, dim=-1).transpose(-1, -2)
        )
        sim = sim - 1e9 * torch.eye(N, device=x.device)

        k_val = min(self.k, N - 1)
        _, idx = sim.topk(k_val, dim=-1)

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
    def __init__(self, dim=256, layers=2):
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
# Token-level SSM Decoder (Mamba-style, lightweight)
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
# Full TSP Model (NO VALUE HEAD)
# ==================================================
class TSPModel(nn.Module):
    def __init__(self, dim=256, layers=4):
        super().__init__()
        self.encoder = GraphEncoder(dim)
        self.decoder = nn.ModuleList([SSMBlock(dim) for _ in range(layers)])
        self.query = nn.Linear(dim, dim)

    def rollout(self, coords, greedy=False):
        B, N, _ = coords.shape
        device = coords.device

        node_emb = self.encoder(coords)

        visited = torch.zeros(B, N, dtype=torch.bool, device=device)
        visited[:, 0] = True

        current = torch.zeros(B, dtype=torch.long, device=device)
        token = node_emb[:, 0:1, :]
        states = [None] * len(self.decoder)

        log_probs = torch.zeros(B, device=device)
        entropies = torch.zeros(B, device=device)
        tour_len = torch.zeros(B, device=device)

        batch_idx = torch.arange(B, device=device)

        for _ in range(N - 1):
            h = token
            for i, layer in enumerate(self.decoder):
                h, states[i] = layer(h, states[i])

            logits = torch.einsum(
                "bd,bnd->bn",
                self.query(h).squeeze(1),
                node_emb
            )
            logits = logits.masked_fill(visited, -1e9)

            dist = Categorical(logits=logits)
            nxt = logits.argmax(dim=-1) if greedy else dist.sample()

            if not greedy:
                log_probs += dist.log_prob(nxt)
                entropies += dist.entropy()

            prev = current
            tour_len += torch.norm(
                coords[batch_idx, prev] -
                coords[batch_idx, nxt],
                dim=-1
            )

            visited = visited.clone()
            visited[batch_idx, nxt] = True
            current = nxt
            token = node_emb[batch_idx, nxt].unsqueeze(1)

        tour_len += torch.norm(
            coords[batch_idx, current] - coords[:, 0],
            dim=-1
        )

        return log_probs, entropies, tour_len

# ==================================================
# Training Loop (PURE TSP LOSS, TPU)
# ==================================================
def train(args):
    device = xm.xla_device()
    model = TSPModel(dim=args.dim, layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    entropy_coef = args.entropy
    B = args.batch
    N = args.n_nodes

    start = time.time()

    for step in range(args.steps):
        coords = torch.rand(B, N, 2, device=device)

        logp, ent, length = model.rollout(coords, greedy=False)

        with torch.no_grad():
            _, _, greedy_len = model.rollout(coords, greedy=True)

        loss = ((length - greedy_len) * logp).mean() - entropy_coef * ent.mean()

        opt.zero_grad()
        loss.backward()
        xm.optimizer_step(opt)

        if step % args.log_every == 0:
            xm.mark_step()
            elapsed = time.time() - start
            print(
                f"[TPU][N={N}] step {step} | "
                f"loss {loss.detach().cpu():.3f} | "
                f"tour {length.mean().detach().cpu():.3f} | "
                f"time {elapsed/60:.2f} min"
            )

# ==================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_nodes", type=int, default=100)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--log_every", type=int, default=50)

    args = parser.parse_args()
    train(args)
