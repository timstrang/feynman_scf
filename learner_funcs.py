import init_funcs as IN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math

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

    return torch.from_numpy(v_mat).float(), torch.from_numpy(prop_mat), cell_dct


class DSet_VP(Dataset):
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
    dset = DSet_VP(params_list, hyper_dct)
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle)


def make_params_dct(
    size,
    max_length,
    valid_moves=IN.std_moveset,
    v_func=IN.get_null_V,
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
# dloader = get_dataloader(params_ls, test_hyper_dct)


def DTree_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


# First, we want to check if this works in the first place.
# We can later utilize this forward process for testing data generation

# v_tens, prop_tens, cell_dct = get_data_tensors(params_ls[-1])
# real = prop_tens.real.numpy()
# v_mat = v_tens.numpy()

shape = [10, 10]
length = shape[0] * shape[1]
# psi_0np = np.random.uniform(-1, 1, length) + 1j * np.random.uniform(-1, 1, length)


def make_gaussian_wavepacket(
    grid_shape=[10, 10],
    width_params=[1, 1],
    momenta=[0, 0],
    mean=[5, 5],
    mass=1,
    hbar=1,
):
    size = grid_shape[0] * grid_shape[1]

    def gaussian_1d(index):
        width = width_params[index]
        momentum = momenta[index]
        mean_x = mean[index]
        pi = 3
        psi_x = (
            lambda x: (2 / pi) ** 0.25
            * width ** (-0.5)
            * np.exp((-(width ** (-2)) + 1j * momentum) * (x - mean_x))
        )
        return np.asarray([psi_x(x) for x in range(grid_shape[index])])

    psi_xy = np.outer(gaussian_1d(0), gaussian_1d(1))
    return psi_xy


psi_0np = make_gaussian_wavepacket()
plt.imshow(psi_0np.real)
plt.show()
psi_0 = torch.from_numpy(psi_0np)


# psi must all be tensors
def evolution(psi_0, prop_tens, N, view=False):
    psi_t = [psi_0]
    for n in range(N):
        psi_cur = psi_t[-1]
        new = prop_tens.matmul(psi_cur)
        psi_t.append(new)
    psi_geom = []
    if view:
        pass
    return psi_t


psi_t = evolution(psi_0, prop_tens, 1)
psi_tnp = [p.numpy() for p in psi_t]
psi_tnp_geom = [p.reshape(shape) for p in psi_tnp]
plt.imshow(psi_tnp_geom[0].real, interpolation="none")
plt.show()
