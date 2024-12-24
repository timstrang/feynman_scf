import init_funcs as IN
import numpy as np
import sklearn.tree as sktree
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

### Number of paths scales geometrically as num_cells * (1 - num_moves**max_length) / (1 - num_moves)


def get_data_tensors(params_dct):
    cell_dct, _, prop_mat = IN.get_prop_mat(**params_dct)

    def get_v_mat(cell_dct):
        sz = params_dct["size"]
        v_mat = np.zeros([sz, sz], dtype=float)
        for cell in cell_dct.values():
            cell_coords = list(cell[0])
            v_mat[cell_coords[0]][cell_coords[1]] = cell[-1]
        return v_mat

    v_mat = get_v_mat(cell_dct)

    return torch.from_numpy(v_mat).float(), torch.from_numpy(prop_mat)


class DSet_vmat_pmat(Dataset):
    def __init__(self, params_list, hyper_dct):
        self.params_list = params_list
        self.num_output_paths = hyper_dct["num_output_paths"]

    def __len__(self):
        return len(self.params_list)

    def __getitem__(self, idx):
        v_tens, prop_tens = get_data_tensors(self.params_list[idx])
        return v_tens, prop_tens


def get_dataloader(params_list, hyper_dct):
    batch_size = hyper_dct["batch_size"]
    shuffle = hyper_dct["shuffle"]
    dset = DSet_vmat_pmat(params_list, hyper_dct)
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle)


def make_params_dct(
    size,
    max_length,
    valid_moves=IN.std_moveset,
    v_func=IN.get_harmonic_V,
    k_func=IN.get_binary_K,
):
    return {
        "size": size,
        "max_length": max_length,
        "valid_moves": valid_moves,
        "v_func": v_func,
        "k_func": k_func,
    }


test_hyper_dct = {
    "num_output_paths": 200,
    "batch_size": 1,
    "shuffle": True,
}
test_size_len = [(10, 3), (10, 4), (10, 7)]
params_ls = [make_params_dct(*sl) for sl in test_size_len]
dloader = get_dataloader(params_ls, test_hyper_dct)

