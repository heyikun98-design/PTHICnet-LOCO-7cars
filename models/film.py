import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.affine = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, h, cond):
        gamma_beta = self.affine(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return gamma * h + beta

