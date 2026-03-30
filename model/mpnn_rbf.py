import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util_fxns import *

class MessageLayer(nn.Module):
    """
    Simple message passing layer
    hidden_dim: int, the latent dimension
    n_rbf: integer, number of rbf's in embedding
    Returns: latent embedding after message passing
    """
    def __init__(self, hidden_dim, n_rbf):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, i, j, rbf):
        m = torch.cat([h[j], rbf], dim=-1)
        m = self.mlp(m)

        out = torch.zeros_like(h)
        out.index_add_(0, i, m)

        return h + out
    
class SimpleMPNN(nn.Module):
    """
    Simple message passing neural network
    n_elements: int, number of element types for Z embedding
    hidden_dim: int, the latent dimension
    n_rbf: integer, number of rbf's in embedding
    cutoff: float, the local atomic environment
    n_layers: number of hidden layers
    cell: simulation cell length (angstrom); cubic assumed
    Returns: latent embedding after message passing
    """
    def __init__(self,
                 n_elements=2,
                 hidden_dim=64,
                 n_rbf=32,
                 cutoff=5.0,
                 n_layers=3,
                 cell=12.42):

        super().__init__()

        self.cell = cell

        self.embedding = nn.Embedding(n_elements, hidden_dim)
        self.rbf = RBF(cutoff, n_rbf)

        self.layers = nn.ModuleList([
            MessageLayer(hidden_dim, n_rbf)
            for _ in range(n_layers)
        ])

        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.energy_head[-1].weight)
        nn.init.zeros_(self.energy_head[-1].bias)

    def forward(self, R, Z, i, j):

        h = self.embedding(Z)

        rij = R[j] - R[i]
        #rij -= self.cell*torch.round(rij/self.cell)
        dist = torch.norm(rij, dim=-1)

        rbf = self.rbf(dist)

        for layer in self.layers:
            h = layer(h, i, j, rbf)

        E = self.energy_head(h).sum()

        return E

