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


spec_gr = [('trj1', float32[:, :, :]), ('n_atoms1', int_), ('n_frames', int_), ('v_shell', float32[:]), ('box_dim', float32[:, :]), ('r_vals', float32[:]), ('dr', float32), ('trj2', float32[:, :, :]), ('n_atoms2', int_)]


@jitclass(spec_gr)
class gr:
    def __init__(self, trj1, n_atoms1, n_frames, v_shell, box_dim, r_vals, trj2=np.zeros((1, 1, 1), dtype=np.float32), n_atoms2=0):
        """
        Args:
            trj1 (numpy array): Trajectory file for type 1 atoms processed into numpy array
            n_atoms1 (int): Number of type 1 atoms
            n_frames (int): Number of frames to sample
            v_shell (numpy array): Array of shell volumes
            box_dim (numpy array): Box dimension array of shape n_frames x 6
            r_vals (numpy array): Array of radii to bin
        """
        self.trj1 = trj1
        self.n_atoms1 = n_atoms1
        self.n_frames = n_frames
        self.v_shell = v_shell
        self.box_dim = box_dim
        self.r_vals = r_vals
        self.trj2 = trj2
        self.n_atoms2 = n_atoms2
        self.dr = 0.005

    def pbc(self, dist, box_len):
        return round(dist / box_len) * box_len

    @property
    def g_AA(self):
        rdf = np.zeros((self.n_frames, len(self.r_vals)))
        for i in range(self.n_frames):
            box_len = np.max(self.box_dim[i, :3])
            box_vol = box_len ** 3
            rho = self.n_atoms1 / box_vol
            for j in range(self.n_atoms1 - 1):
                for k in range(j + 1, self.n_atoms1):
                    dx = self.trj1[i, j, 0] - self.trj1[i, k, 0]
                    dx -= self.pbc(dx, box_len)
                    dy = self.trj1[i, j, 1] - self.trj1[i, k, 1]
                    dy -= self.pbc(dy, box_len)
                    dz = self.trj1[i, j, 2] - self.trj1[i, k, 2]
                    dz -= self.pbc(dz, box_len)
                    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    if r < self.r_vals[-1]:
                        idx = int(r / self.dr)
                        rdf[i, idx] += 2
            rdf[i] /= (rho * self.n_atoms1 * self.v_shell)
        print("Done!")
        return rdf

    @property
    def g_AB(self):
        rdf = np.zeros((self.n_frames, len(self.r_vals)))
        for i in range(self.n_frames):
            box_len = np.max(self.box_dim[i, :3])
            box_vol = box_len ** 3
            rho = self.n_atoms2 / box_vol
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
                        rdf[i, idx] += 2
            rdf[i] /= (rho * self.n_atoms2 * self.v_shell)
        print("Done!")
        return rdf
