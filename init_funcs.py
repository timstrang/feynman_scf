import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import dev_util as util
import imageio


def harmonic_well_potential(pt, center, lin_normalize_factor=1):
    if len(pt) != len(center):
        raise ValueError("dimension mismatch")
    radial_dist = np.linalg.norm(pt - center) * lin_normalize_factor
    return radial_dist**2


def linear_kinetic_energy(p1, p2, mass=1, lin_scale_factor=1, dt=1):
    radial_dist = np.linalg.norm(p1 - p2) / lin_scale_factor
    velocity = radial_dist / dt
    return float(0.5 * mass * velocity**2)


harmonic_vfunc = lambda p: harmonic_well_potential(
    p, [25, 25], lin_normalize_factor=(1e-7) * 3
)
blank_vfunc = lambda p: 0


def gaussian_1d(mean, width, momentum, length):
    pi = math.pi
    psi_x = (
        lambda x: (2 / pi) ** 0.25
        * width ** (-0.5)
        * np.exp(-(width ** (-2)) * (x - mean) * (x - mean) / 2)
        * np.exp(1j * momentum * x)
    )
    return np.asarray([psi_x(x) for x in range(length)])


def gaussian_nd(means, widths, momenta, shape):
    dim_components = []
    for param_tuple in zip(means, widths, momenta, shape):
        dim_components.append(gaussian_1d(*param_tuple))
    meshgrid = np.meshgrid(*dim_components, indexing="ij")
    psi_nd = meshgrid[0]
    for g in meshgrid[1:]:
        psi_nd = psi_nd * g
    return psi_nd


class nDim_Grid:
    def __init__(self, size):
        self.size = size
        self.dimension = len(size)
        matrix_sides = []
        for side_len in size:
            matrix_sides.append(range(side_len))
        meshgrid = np.meshgrid(*matrix_sides)
        self.spatial_grid = np.stack([*meshgrid], axis=-1)
        self.shape = self.spatial_grid.shape
        self.vectorized = self.spatial_grid.reshape(-1, self.dimension).astype(int)
        self.arr_side = len(self.vectorized)
        unit_vectors = []
        for dim in range(self.dimension):
            unit_vector = np.zeros(self.dimension)
            unit_vector[dim] = 1
            unit_vectors.append(unit_vector)
            unit_vectors.append(-1 * unit_vector)
        # secondary_vectors = []
        # for v1 in unit_vectors:
        #     for v2 in unit_vectors:
        #         secondary_vectors.append(v1 + v2)
        self.adjacency_def = (
            unit_vectors  # + [s for s in secondary_vectors if s not in unit_vectors]
        )
        neighborhoods = []
        for pt in self.vectorized:
            nghbrs = [pt + unit_vec for unit_vec in self.adjacency_def]
            coded_nghbrs = [self.vectorized_index_by_coordinate(pt) for pt in nghbrs]
            neighborhoods.append([code for code in coded_nghbrs if code != None])
        self.neighborhoods = neighborhoods

    def vectorized_index_by_coordinate(self, pt):
        pt = pt.astype(int)
        if np.min(pt) >= 0 and np.min(self.size - pt) > 0:
            for idx, coord in enumerate(self.vectorized):
                if np.array_equal(pt, coord):
                    return idx
        return None

    def init_path_array(self):
        shape_vec = [self.arr_side, self.arr_side, 1]
        path_array = np.empty(shape_vec, dtype=object)
        for i in range(self.arr_side):
            for j in range(self.arr_side):
                path_array[i][j][0] = []
        return path_array


class Taxicab_Action_Grid(nDim_Grid):
    def __init__(self, grid, vfunc, kfunc, hbar=1, N=3):
        self.__dict__.update(grid.__dict__)
        self.N = N
        self.hbar = hbar
        # vfunc must take an n-element np vector
        self.vfunc = vfunc
        self.flat_V = np.asarray([vfunc(p) for p in self.vectorized])
        # kfunc must take as inputs two n-dimensional points as np arrays
        self.kfunc = kfunc
        kinetic_transition_energy = []
        for i in range(self.arr_side):
            start_cell = self.vectorized[i]
            energy = [
                self.kfunc(start_cell, self.vectorized[nb])
                for nb in self.neighborhoods[i]
            ]
            kinetic_transition_energy.append(energy)
        self.flat_K = kinetic_transition_energy
        self.path_array = None
        self.paths_of_length_n(N)
        self.action_mat = self.get_action_mat()
        prop_mat = np.exp(self.action_mat * (0 + 1j) / self.hbar)
        for i in range(self.arr_side):
            for j in range(self.arr_side):
                if abs(i - j) > self.N:
                    prop_mat[i, j] = 0
        self.prop_mat = prop_mat

    def paths_of_length_n(self, N):
        def path_extension(path_set, k_set, v_set):
            new_paths = []
            new_k = []
            new_v = []
            for path, k_val, v_val in zip(path_set, k_set, v_set):
                start = path[0]
                end = path[-1]
                for nghbr, kenergy in zip(self.neighborhoods[end], self.flat_K[end]):
                    new_paths.append(path + [nghbr])
                    new_k.append(kenergy + k_val)
                    new_v.append(self.flat_V[nghbr] + v_val)
            return new_paths, new_k, new_v

        # this if/else initializes the path building routine
        # first index is start cell, second is end
        if not isinstance(self.path_array, np.ndarray):
            prev_progress = 1
            prev_path_layer = self.init_path_array()
            prev_k_layer = self.init_path_array()
            prev_v_layer = self.init_path_array()
            for i in range(self.arr_side):
                for j in range(self.arr_side):
                    prev_path_layer[i, j, 0] += [[i]]
                    prev_k_layer[i, j, 0] += [0]
                    prev_v_layer[i, j, 0] += [self.flat_V[i]]
        else:
            prev_progress = self.path_array.shape[-1] + 1
            prev_path_layer = self.path_array[:, :, -1:]
            prev_k_layer = self.k_array[:, :, -1:]
            prev_v_layer = self.v_array[:, :, -1:]

        # Proper path building routine
        for n in tqdm(range(N + 1)):
            if n <= prev_progress:
                continue
            next_path_layer = self.init_path_array()
            next_k_layer = self.init_path_array()
            next_v_layer = self.init_path_array()
            for i in range(self.arr_side):
                for j in range(self.arr_side):
                    old_paths = prev_path_layer[i][j][0]
                    old_k = prev_k_layer[i][j][0]
                    old_v = prev_v_layer[i][j][0]
                    new_paths, new_k, new_v = path_extension(old_paths, old_k, old_v)
                    for path, kenergy, venergy in zip(new_paths, new_k, new_v):
                        coords = [path[0], path[-1]]
                        next_path_layer[*coords, 0] += [path]
                        next_k_layer[*coords, 0] += [kenergy]
                        next_v_layer[*coords, 0] += [venergy]
                    if n == 2:
                        break

            if not isinstance(self.path_array, np.ndarray):
                self.path_array = next_path_layer
                self.k_array = next_k_layer
                self.v_array = next_v_layer
            else:
                self.path_array = np.concatenate(
                    [self.path_array, next_path_layer], axis=-1
                )
                self.k_array = np.concatenate([self.k_array, next_k_layer], axis=-1)
                self.v_array = np.concatenate([self.v_array, next_v_layer], axis=-1)
            prev_path_layer = next_path_layer
            prev_k_layer = next_k_layer

    def get_action_mat(self):
        action_mat = np.zeros([self.arr_side, self.arr_side])
        path_arr = self.path_array
        k_arr = self.k_array
        v_arr = self.v_array
        for i in range(self.arr_side):
            for j in range(self.arr_side):
                num_paths = 0
                for n in range(path_arr.shape[-1]):
                    k_sum = sum(k_arr[i, j, n]) * (n + 2)
                    v_sum = sum(v_arr[i, j, n]) / (n + 2)
                    action_mat[i][j] += k_sum - v_sum
                    num_paths += len(path_arr[i, j, n])

        return np.array(action_mat, dtype=np.complex128)

    def visualize_prop_mat(
        self, real=True, imaginary=False, savefile="unnamed_prop_mat"
    ):
        if real:
            plt.imshow(self.prop_mat.real)
            plt.savefig(savefile)
            print(f"Saved propagation matrix image to {savefile}")
            plt.close()

    def forward(self, wavefunc, steps=1):
        nxt_wavefunc = wavefunc.reshape(self.arr_side)
        for i in range(steps):
            nxt_wavefunc = np.matmul(self.prop_mat, nxt_wavefunc)
        return nxt_wavefunc.reshape(wavefunc.shape)


class Straightline_Action_Grid(nDim_Grid):
    def __init__(self, grid, vfunc, kfunc, hbar=1):
        self.__dict__.update(grid.__dict__)
        self.hbar = hbar
        # vfunc must take an n-element np vector
        self.vfunc = vfunc
        self.flat_V = np.asarray([vfunc(p) for p in self.vectorized])
        # kfunc must take as inputs two n-dimensional points as np arrays
        self.kfunc = kfunc
        k_mat = np.zeros([self.arr_side, self.arr_side])
        v_mat = np.zeros([self.arr_side, self.arr_side])
        for i in range(self.arr_side):
            for j in range(self.arr_side):
                coord_i = self.vectorized[i]
                coord_j = self.vectorized[j]
                k_mat[i, j] += self.kfunc(coord_i, coord_j)
                v_mat[i, j] += (self.vfunc(coord_i) + self.vfunc(coord_j)) / 2
        self.k_mat = k_mat
        self.v_mat = v_mat
        self.action_mat = self.get_action_mat()
        self.prop_mat = np.exp(self.action_mat * (0 + 1j) / self.hbar)

    def get_action_mat(self):
        action_mat = self.k_mat - self.v_mat
        return np.array(action_mat, dtype=np.complex128)

    def visualize_prop_mat(
        self, real=True, imaginary=False, savefile="unnamed_prop_mat"
    ):
        if real:
            plt.imshow(self.prop_mat.real)
            plt.savefig(savefile)
            print(f"Saved propagation matrix image to {savefile}")
            plt.close()

    def forward(self, wavefunc, steps=1):
        nxt_wavefunc = wavefunc.reshape(self.arr_side)
        for i in range(steps):
            nxt_wavefunc = np.matmul(self.prop_mat, nxt_wavefunc)
        return nxt_wavefunc.reshape(wavefunc.shape)


gauss_dct = {
    "means": [250],
    "widths": [25],
    "momenta": [0],
    "shape": [500],
}

grid = nDim_Grid([500])
# sgrid = Taxicab_Action_Grid(grid, blank_vfunc, linear_kinetic_energy, hbar=100, N=50)
sgrid = Straightline_Action_Grid(grid, blank_vfunc, linear_kinetic_energy, hbar=100)
sgrid.visualize_prop_mat(savefile="test")
psi0 = gaussian_nd(*gauss_dct.values())


def make_evolution_gif_1d(action_grid, psi0, frames, folder="temp_gif"):
    util.clear_folder(folder)
    length = len(psi0)
    x = np.arange(length)
    files = []
    plt.plot(x, psi0.real / sum(psi0.real))
    file = f"{folder}/psi0"
    files.append(file)
    plt.savefig(file)
    plt.close()
    psi = psi0
    for i in range(frames):
        psi = sgrid.forward(psi)
        plt.plot(x, psi.real / sum(psi.real))
        file = f"{folder}/psi{i+1}"
        plt.savefig(file)
        files.append(file)
        plt.close()
    with imageio.get_writer("psi_t.gif", mode="I") as writer:
        for file in files:
            image = imageio.imread(file + ".png")
            writer.append_data(image)


make_evolution_gif_1d(sgrid, psi0, 30)
