import numpy as np
import matplotlib.pyplot as plt

### cell_dct, fully constructed, has the format id: [x, adj, v]


std_moveset = [np.asarray(move) for move in [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]]


def array2dct_ind(array, cell_dct):
    keys = cell_dct.keys()
    if isinstance(array, int) and array in keys:
        return array
    elif np.issubdtype(array.dtype, np.integer):
        ls_raw = list(array)
        j = 0
        while j < len(ls_raw) and ls_raw[j] == 0:
            j += 1
        ls = ls_raw[j:]
        word = ""
        for i in ls:
            if i < 0:
                return None
            word += str(i)
        if word == "":
            return 0
        elif int(word) in cell_dct.keys():
            return int(word)
    return None


def dict_ind2array(index, grid_shape):
    b = grid_shape[1]
    return [index // b, n % b]


def cond_list2array(obj):
    if isinstance(obj, list):
        obj = np.asarray(obj)
    return obj


def generate_cell_dct(size, valid_moves, v_func, k_func):
    centers = []
    for i in range(size):
        for j in range(size):
            centers.append(np.asarray([i, j]))
    cell_dct = {ind: [val] for ind, val in enumerate(centers)}

    for cell_id, pt in cell_dct.items():
        pt = pt[0]
        pt_adj = []
        for move in valid_moves:
            index = array2dct_ind(move + pt, cell_dct)
            if index != None:
                pt_adj.append(index)
        cell_dct[cell_id] += [pt_adj]

    for cell_id in cell_dct.keys():
        cell_dct[cell_id] += [v_func(cell_dct, cell_id)]
        cell_dct[cell_id][1] = k_func(cell_dct, cell_id)
    return cell_dct


def get_radial_dist(cell_dct, cell_id):
    center_vectors = [cell_dct[j][0] for j in cell_dct.keys()]
    return np.linalg.norm(cell_dct[cell_id][0] - np.mean(center_vectors, axis=0))


def get_harmonic_V(cell_dct, cell_id):
    r = get_radial_dist(cell_dct, cell_id)
    return r**2


def get_null_V(cell_dct, cell_id):
    return 0


def get_binary_K(cell_dct, cell_id, time_step=1, space_step=1):
    adj = cell_dct[cell_id][1]
    k_calc = lambda start, target: (
        0.5 * (space_step / time_step) ** 2 if start != target else 0
    )
    return {targ: k_calc(cell_id, targ) for targ in adj}


def generate_path_vectors(cell_dct, max_length):
    paths = {cell_id: [[cell_id]] for cell_id in cell_dct.keys()}
    tmp_paths_prev = [[cell_id] for cell_id in cell_dct.keys()]
    length = 2
    while length <= max_length:
        tmp_paths_nxt = []
        for pth in tmp_paths_prev:
            end = pth[-1]
            for adj_cell in cell_dct[end][1].keys():
                pth_extended = pth + [adj_cell]
                tmp_paths_nxt += [pth_extended]
                paths[adj_cell] += [pth_extended]
        tmp_paths_prev = tmp_paths_nxt.copy()
        length += 1
    return paths


def add_S_to_paths(cell_dct, path_dct, hbar=1):
    def s_calc(path):
        factor = len(path)
        s = 0
        for i, cell_id in enumerate(path[:-1]):
            cell = cell_dct[cell_id]
            k = cell[1][path[i + 1]] * factor
            v = cell[2] / factor
            s += k - v
        s -= cell_dct[path[-1]][2] / factor
        return s / hbar

    return {key: [[pth, s_calc(pth)] for pth in val] for key, val in path_dct.items()}


def make_propagation_matrix(cell_dct, path_dct, hbar=1):
    action_paths = add_S_to_paths(cell_dct, path_dct, hbar=hbar)
    num_cells = len(cell_dct.keys())
    prop_mat = np.zeros([num_cells, num_cells], dtype=np.complex128)
    for key, val in action_paths.items():
        row = key
        row_vec = np.zeros(num_cells, dtype=np.complex128)
        for pth, s in val:
            row_vec[pth[0]] += np.exp(s * (0 + 1j))
        prop_mat[row] = row_vec
    return prop_mat


def get_prop_mat(
    size,
    max_length,
    valid_moves=std_moveset,
    v_func=get_harmonic_V,
    k_func=get_binary_K,
    hbar=1,
):
    cell_dct = generate_cell_dct(size, valid_moves, v_func, k_func)
    path_dct = generate_path_vectors(cell_dct, max_length)
    prop_mat = make_propagation_matrix(cell_dct, path_dct, hbar=hbar)
    return cell_dct, path_dct, prop_mat
