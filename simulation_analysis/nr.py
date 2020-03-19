"""
@author Kevin Juan
Analyze GROMACS trajectory files for water/ice
Dependencies:
- MDAnalysis
- numba
- numpy
"""

from MDAnalysis import*
import numpy as np
from numba import jitclass
from numba import float32, int_


spec_nr = [('trj1', float32[:, :, :]), ('n_atoms1', int_), ('n_samples', int_), ('box_dim', float32[:, :]), ('r_vals', float32[:]), ('dr', float32), ('trj2', float32[:, :, :]), ('n_atoms2', int_)]


@jitclass(spec_nr)
class nr:
    def __init__(self, trj1, n_atoms1, n_samples, box_dim, r_vals, trj2=np.zeros((1, 1, 1), dtype=np.float32), n_atoms2=0):
        self.trj1 = trj1
        self.n_atoms1 = n_atoms1
        self.n_samples = n_samples
        self.box_dim = box_dim
        self.r_vals = r_vals
        self.trj2 = trj2
        self.n_atoms2 = n_atoms2
        self.dr = 0.005

    def pbc(self, dist, box_len):
        return round(dist / box_len) * box_len

    @property
    def n_AA(self):
        nr = np.zeros((self.n_samples, len(self.r_vals)))
        for i in range(self.n_samples):
            box_len = np.max(self.box_dim[i, :3])
            for j in range(self.n_atoms1):
                for k in range(self.n_atoms1):
                    if j != k:
                        dx = self.trj1[i, j, 0] - self.trj1[i, k, 0]
                        dx -= self.pbc(dx, box_len)
                        dy = self.trj1[i, j, 1] - self.trj1[i, k, 1]
                        dy -= self.pbc(dy, box_len)
                        dz = self.trj1[i, j, 2] - self.trj1[i, k, 2]
                        dz -= self.pbc(dz, box_len)
                        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                        if r < self.r_vals[-1]:
                            idx = int(r / self.dr)
                            nr[i, idx] += 1
        nr /= self.n_atoms1
        print("Done!")
        return nr

    @property
    def n_AB(self):
        nr = np.zeros((self.n_samples, len(self.r_vals)))
        for i in range(self.n_samples):
            box_len = np.max(self.box_dim[i, :3])
            for j in range(self.n_atoms1):
                for k in range(self.n_atoms2):
                    dx = self.trj1[i, j, 0] - self.trj2[i, k, 0]
                    dx -= self.pbc(dx, box_len)
                    dy = self.trj1[i, j, 1] - self.trj2[i, k, 1]
                    dy -= self.pbc(dy, box_len)
                    dz = self.trj1[i, j, 2] - self.trj2[i, k, 2]
                    dz -= self.pbc(dz, box_len)
                    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    if r < self.r_vals[-1]:
                        idx = int(r / self.dr)
                        nr[i, idx] += 1
        nr /= self.n_atoms1
        print("Done!")
        return nr
