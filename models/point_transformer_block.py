import torch
import torch.nn as nn


class PointTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.theta = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.phi = nn.Linear(dim, dim)
        self.psi = nn.Linear(dim, dim)
        self.alpha = nn.Linear(dim, dim)
        self.gamma = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x_i, x_j, rel_pos):
        # x_i: [B, N, C], x_j: [B, N, K, C], rel_pos: [B, N, K, 3]
        delta_pos = self.theta(rel_pos)
        q = self.phi(x_i).unsqueeze(2)
        k = self.psi(x_j)
        v = self.alpha(x_j) + delta_pos
        attn = self.gamma(q - k + delta_pos)
        attn = torch.softmax(attn, dim=2)
        out = torch.sum(attn * v, dim=2)
        out = self.norm(x_i + out)
        out = self.norm(out + self.ffn(out))
        return out

