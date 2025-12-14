import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.encoder import StaticGraphEncoder
from src.core.ssm import StateSpaceTokenBlock
from src.core.moe import MoESSMBlock


class SSMMoETSP(nn.Module):
    """
    End-to-end TSP model:

    Encoder (static):
        coords -> StaticGraphEncoder -> node embeddings

    Decoder (autoregressive):
        token -> [MoE(StateSpaceTokenBlock)] x L -> query -> logits over nodes

    This model is:
    - Autoregressive
    - State-space (no self-attention in decoder)
    - Sparse (kNN encoder + MoE decoder)
    """

    def __init__(
        self,
        model_dim: int = 128,
        enc_layers: int = 3,
        dec_layers: int = 3,
        num_heads: int = 8,
        knn_k: int = 16,
        max_nodes: int = 256,
        num_experts: int = 4,
        moe_top_k: int = 1,
    ):
        super().__init__()

        # -------- Encoder --------
        self.encoder = StaticGraphEncoder(
            model_dim=model_dim,
            num_layers=enc_layers,
            num_heads=num_heads,
            knn_k=knn_k,
            max_nodes=max_nodes,
        )

        # -------- Decoder (MoE-SSM stack) --------
        self.decoder_layers = nn.ModuleList([
            MoESSMBlock(
                expert_cls=StateSpaceTokenBlock,
                dim=model_dim,
                num_experts=num_experts,
                top_k=moe_top_k,
            )
            for _ in range(dec_layers)
        ])

        # Project token to query space
        self.query_proj = nn.Linear(model_dim, model_dim)

    # -------------------------------------------------
    # Encode graph (static)
    # -------------------------------------------------
    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (B, N, 2)
        returns: node_embeddings (B, N, D)
        """
        return self.encoder(coords)

    @torch.no_grad()
    def greedy_decode(self, coords: torch.Tensor):
        """
        Greedy rollout for a single TSP instance.

        coords: (1, N, 2)
        returns:
            route: list[int]
            total_length: float
        """
        device = coords.device
        B, N, _ = coords.shape
        assert B == 1, "greedy_decode supports batch size 1 only"

        node_emb = self.encode(coords)

        visited = torch.zeros(N, dtype=torch.bool, device=device)
        visited[0] = True

        token = node_emb[:, 0:1, :]  # start at node 0

        # init decoder states
        states = [[None] * len(self.decoder_layers[0].experts)
                  for _ in range(len(self.decoder_layers))]

        route = [0]
        total_dist = 0.0
        cur = 0

        for _ in range(N - 1):
            logits, states, _ = self.decode_step(token, node_emb, states)

            logits = logits.squeeze(0)
            logits = logits.view(-1)
            mask = visited.bool()
            logits = logits.masked_fill(mask, -1e9)

            nxt = logits.argmax().item()

            # distance
            total_dist += torch.norm(
                coords[0, cur] - coords[0, nxt]
            ).item()

            visited[nxt] = True
            route.append(nxt)
            cur = nxt

            token = node_emb[:, nxt:nxt+1, :]

        # return to start
        total_dist += torch.norm(
            coords[0, cur] - coords[0, 0]
        ).item()
        route.append(0)

        return route, total_dist

    # -------------------------------------------------
    # Decode one step
    # -------------------------------------------------
    def decode_step(self, token, node_emb, states):
        """
        token   : (B, 1, D)
        node_emb: (B, N, D)
        states  : list of per-layer expert states

        returns:
            logits: (B, N)
            next_states
            lb_loss (load balancing)
        """
        lb_loss = 0.0
        h = token
        next_states = []

        for layer, layer_states in zip(self.decoder_layers, states):
            h, new_states, lb = layer(h, layer_states)
            lb_loss = lb_loss + lb
            next_states.append(new_states)

        # Query -> logits over nodes
        q = self.query_proj(h).squeeze(1)  # (B, D)
        logits = torch.matmul(q, node_emb.transpose(1, 2)) / math.sqrt(q.size(-1))

        return logits, next_states, lb_loss


# ======================================================
# SIMPLE SMOKE TEST (NO TRAINING)
# ======================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    coords = torch.rand(1, 10, 2)

    model = SSMMoETSP(
        model_dim=128,
        enc_layers=3,
        dec_layers=2,
        num_experts=4,
    )

    model.eval()

    route, length = model.greedy_decode(coords)

    print("Route:", route)
    print("Tour length:", round(length, 4))
