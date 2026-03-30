import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

from model.mpnn_rbf import *
from util_fxns import *

path='/u/scratch/w/wborrell/solvated_electron_dft/'
n_atoms = 384
n_tr = 500

# training data
# cds are (( n_steps, 1152 )) where 1152 is flattened 384 atoms x 3 axes
cds_tr = np.load(path+'train/set.001/coord.npy')
# scalar 
es_tr = np.load(path+'train/set.001/energy.npy')
# same as cds
frc_tr = np.load(path+'train/set.001/force.npy')
# (( n_steps, 9 )) where 9 is flattened cell primitives
cell_tr = np.load(path+'train/set.001/box.npy')
int_types_tr = np.loadtxt(path+'train/type.raw').astype('int')
syms_tr = list(np.loadtxt(path+'train/type_map.raw', dtype=str, delimiter=','))
sym_types_tr = map_types(int_types_tr,syms_tr)

Zs_tr = torch.tensor(int_types_tr, dtype=torch.int32)
es_tr = torch.tensor(es_tr, dtype=torch.float32)
cdsf_tr = torch.tensor(cds_tr.reshape((cds_tr.shape[0], n_atoms, 3)), dtype=torch.float32)
frcf_tr = torch.tensor(frc_tr.reshape((frc_tr.shape[0], n_atoms, 3)), dtype=torch.float32)

dataset_tr = MDDataset(cdsf_tr[0:n_tr], frcf_tr[0:n_tr], es_tr[0:n_tr], Zs_tr)
E_mean_tr = dataset_tr.E.mean()
dataset_tr.E -= E_mean_tr
loader_tr = DataLoader(dataset_tr,batch_size=None,shuffle=True)

# validation data
# cds are (( n_steps, 1152 )) where 1152 is flattened 384 atoms x 3 axes
cds_val = np.load(path+'val/set.001/coord.npy')
# scalar 
es_val = np.load(path+'val/set.001/energy.npy')
# same as cds
frc_val = np.load(path+'val/set.001/force.npy')
# (( n_steps, 9 )) where 9 is flattened cell primitives
cell_val = np.load(path+'val/set.001/box.npy')
int_types_val = np.loadtxt(path+'val/type.raw').astype('int')
syms_val = list(np.loadtxt(path+'val/type_map.raw', dtype=str, delimiter=','))
sym_types_val = map_types(int_types_val,syms_val)

Zs_val = torch.tensor(int_types_val, dtype=torch.int32)
es_val = torch.tensor(es_val, dtype=torch.float32)
cdsf_val = torch.tensor(cds_val.reshape((cds_val.shape[0], n_atoms, 3)), dtype=torch.float32)
frcf_val = torch.tensor(frc_val.reshape((frc_val.shape[0], n_atoms, 3)), dtype=torch.float32)

dataset_val = MDDataset(cdsf_val[0:n_tr], frcf_val[0:n_tr], es_val[0:n_tr], Zs_val)
E_mean_val = dataset_val.E.mean()
dataset_val.E -= E_mean_val
loader_val = DataLoader(dataset_val,batch_size=None,shuffle=True)


## MAIN

cutoff = 6.0

model = SimpleMPNN(n_elements=2,
                 hidden_dim=32,
                 n_rbf=16,
                 cutoff=cutoff,
                 n_layers=2,
                 cell=12.42)

print('Starting training...',flush=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print_every=100
step = 0
epoch_losses_tr = []
epoch_losses_val = []
for epoch in range(10):
    epoch_loss = 0
    counter=0
    for R, Z, E_ref, F_ref in loader_tr:

        R = R
        Z = Z
        F_ref = F_ref
        E_ref = E_ref

        with torch.no_grad():
            i, j = neighbor_list(R, cutoff, 12.42)

        E_pred, F_pred = energy_forces(model, R, Z, i, j)

        loss = loss_fn(E_pred, E_ref, F_pred, F_ref)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        if step % print_every == 0:
            print(f"step {step:6d} | loss = {loss.item():.6f}",flush=True)
        step+=1
        counter+=1
    val_loss = evaluate(model, loader_val, cutoff, 12.42)
    epoch_losses_tr.append(epoch_loss/counter)
    epoch_losses_val.append(val_loss)
    print(f"epoch {epoch:4d} | train_loss = {loss.item():.6f} | val_loss = {val_loss:.6f}",flush=True)
