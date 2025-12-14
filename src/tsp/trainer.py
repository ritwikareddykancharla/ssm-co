import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.tsp.model import SSMMoETSP


class SimpleTSPTrainer:
    """
    Minimal REINFORCE trainer for TSP.

    - No baseline network
    - Uses tour length as reward
    - Includes MoE load-balancing loss

    This is for SANITY + FIRST LEARNING, not SOTA yet.
    """

    def __init__(
        self,
        model: SSMMoETSP,
        lr: float = 1e-4,
        moe_lb_coef: float = 0.01,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.moe_lb_coef = moe_lb_coef

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def sample_batch(self, batch_size: int, n_nodes: int):
        """Generate random TSP batch."""
        return torch.rand(batch_size, n_nodes, 2, device=self.device)

    def rollout(self, coords: torch.Tensor):
        """
        Perform stochastic rollout for a batch.

        returns:
            log_probs: (B,)
            tour_lengths: (B,)
            moe_loss: scalar
        """
        B, N, _ = coords.shape

        node_emb = self.model.encode(coords)

        visited = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        visited[:, 0] = True

        token = node_emb[:, 0:1, :]

        # init decoder states
        states = [
            [None] * len(self.model.decoder_layers[0].experts)
            for _ in range(len(self.model.decoder_layers))
        ]

        log_probs = torch.zeros(B, device=self.device)
        tour_lengths = torch.zeros(B, device=self.device)
        moe_loss_total = 0.0

        cur = torch.zeros(B, dtype=torch.long, device=self.device)

        for _ in range(N - 1):
            logits, states, lb = self.model.decode_step(token, node_emb, states)
            moe_loss_total = moe_loss_total + lb

            logits = logits.masked_fill(visited, -1e9)
            dist = Categorical(logits=logits)
            nxt = dist.sample()

            log_probs += dist.log_prob(nxt)

            step_dist = torch.norm(
                coords[torch.arange(B), cur] - coords[torch.arange(B), nxt], dim=1
            )
            tour_lengths += step_dist

            visited[torch.arange(B), nxt] = True
            cur = nxt
            token = node_emb[torch.arange(B), nxt].unsqueeze(1)

        # return to depot
        tour_lengths += torch.norm(
            coords[torch.arange(B), cur] - coords[torch.arange(B), 0], dim=1
        )

        return log_probs, tour_lengths, moe_loss_total / B

    def train_step(self, batch_size: int, n_nodes: int):
        self.model.train()

        coords = self.sample_batch(batch_size, n_nodes)
        log_probs, tour_lengths, moe_loss = self.rollout(coords)

        # REINFORCE loss (minimize length)
        loss = (tour_lengths * log_probs).mean()
        loss = loss + self.moe_lb_coef * moe_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "avg_tour": tour_lengths.mean().item(),
            "moe_lb": moe_loss.item(),
        }


# ======================================================
# QUICK TRAINING SANITY TEST
# ======================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SSMMoETSP(
        model_dim=128,
        enc_layers=3,
        dec_layers=2,
        num_experts=4,
    )

    trainer = SimpleTSPTrainer(model, device=device)

    for step in range(10):
        stats = trainer.train_step(batch_size=16, n_nodes=10)
        print(f"Step {step}:", stats)
