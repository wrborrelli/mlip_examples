import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

def map_types(int_types, sym_map):
    conv_dict = {i:str(j) for i,j in enumerate(sym_map)}
    return np.array([conv_dict[k] for k in int_types])

# get Z from atomic symbol
def zfs(sym):
    atomic_dict = {'H':1,'O':6}
    return atomic_dict[sym]

class MDDataset(Dataset):
    def __init__(self, positions, forces, energies, Z):

        self.R = positions.float()
        self.F = forces.float()
        self.E = energies.float()
        self.Z = Z.long()

    def __len__(self):
        return self.R.shape[0]

    def __getitem__(self, idx):
        return (
            self.R[idx],   # (N,3)
            self.Z,        # (N,)
            self.E[idx],   # scalar
            self.F[idx]    # (N,3)
        )

def neighbor_list(R, cutoff, L):
    """
    Vectorized neighbor list with minimum image convention (cubic box)
    R: (N,3) atomic positions, torch float tensor
    cutoff: float defining the local atomic environment
    L: float, box length
    Returns: i_list, j_list (1D tensors of neighbors)
    """
    with torch.no_grad():
        N = R.shape[0]

        Rij = R.unsqueeze(1) - R.unsqueeze(0)
        Rij -= L * torch.round(Rij / L)

        D = torch.norm(Rij, dim=-1)

        mask = (D < cutoff)
        mask.fill_diagonal_(False)

        i_list, j_list = torch.nonzero(mask, as_tuple=True)

    return i_list, j_list

class RBF(nn.Module):
    """
    Simple radial basis function position-based embedding 
    cutoff: float, the local atomic environment
    n_rbf: integer, number of rbf's in embedding
    Returns: i_list, j_list (1D tensors of neighbors)
    """
    def __init__(self, cutoff, n_rbf):
        super().__init__()
        centers = torch.linspace(0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.register_buffer("gamma", torch.tensor(10.0))
        self.cutoff = cutoff

    def forward(self, d):
        rbf = torch.exp(-self.gamma * (d[..., None] - self.centers)**2)
        cutoff = 0.5 * (torch.cos(torch.pi * d / self.cutoff) + 1.0)
        cutoff = cutoff * (d < self.cutoff)
        return rbf * cutoff[..., None]
    
def energy_forces(model, R, Z, i, j):
    """
    Compute energies and forces from the model
    model: pytorch model, neural network capable of predicting local potential energies
    R: float, atomic positions tensor
    Z: int, atomic number (Z)
    i: atom i in pair i,j
    j: atom j in pair i,j
    Returns: energy, force of configuration
    """
    R.requires_grad_(True)

    E = model(R, Z, i, j)

    F = -torch.autograd.grad(
        E, R,
        create_graph=True
    )[0]

    return E, F

def loss_fn(E_pred, E_ref, F_pred, F_ref, force_weight=10):
    """
    Compute energy + force loss function
    E_pred: predicted energy
    E_ref: reference energy
    F_pred: predicted force
    F_ref: reference force
    force_weight: weighting for force error term
    Returns: loss function value
    """
    n_atoms = F_ref.shape[0]

    loss = ((E_pred - E_ref)/n_atoms).pow(2)
    loss += force_weight * (F_pred - F_ref).pow(2).mean()

    return loss
