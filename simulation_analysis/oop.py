"""
@author Kevin Juan
Orientational order parameters
Dependencies:
- MDAnalysis
- numpy
- scipy
"""

from MDAnalysis import*
from MDAnalysis import lib
import numpy as np
from scipy.special import sph_harm


class orientational_op:
    def __init__(self, trj, frame, box_dim, bound, probe_vol):
        """
        Args:
            trj (numpy array): Trajectory file processed into numpy array
            frame (int): Relative trajectory frame number
            box_dim (numpy array): Box dimension array of shape n_frames x 6
            bound (float): Cut off radius in nm
            probe_vol (numpy array): Array of particles within the probe volume
        """
        self.trj = trj
        self.frame = frame
        self.box_dim = box_dim
        self.bound = bound
        self.probe_vol = probe_vol

    def nn(self):
        search = lib.nsgrid.FastNS(self.bound, self.trj[self.frame], self.box_dim[self.frame], pbc=True)
        nn_idx = search.search(self.trj[self.frame, self.probe_vol]).get_indices()
        nn_idx = np.array([np.array(x) for x in nn_idx if x != []])
        nn_dist = search.search(self.trj[self.frame, self.probe_vol]).get_distances()
        nn_dist = np.array([np.array(x) for x in nn_dist if x != []])

        if len(nn_dist.shape) == 1:
            for i in range(len(self.probe_vol)):
                loc = np.where(nn_dist[i] == 0.0)
                nn_idx[i] = np.delete(nn_idx[i], loc[0][0])
                nn_dist[i] = np.delete(nn_dist[i], loc[0][0])
        else:
            loc = np.where(nn_dist[0] == 0.0)
            nn_dist_new = np.delete(nn_dist[0], loc[0][0])
            nn_idx_new = np.delete(nn_idx[0], loc[0][0])
            for i in range(1, len(nn_dist)):
                loc = np.where(nn_dist[i] == 0.0)
                nn_dist_new = np.vstack((nn_dist_new, np.delete(nn_dist[i], loc[0][0])))
                nn_idx_new = np.vstack((nn_idx_new, np.delete(nn_idx[i], loc[0][0])))
            nn_idx = nn_idx_new
            nn_dist = nn_dist_new
        return nn_idx, nn_dist

    def nn_coords(self, ref_atom):
        nn_idx, _ = self.nn()
        coords = self.trj[self.frame, nn_idx[ref_atom]]
        return coords

    def pbc(self, dist):
        max_len = max(self.box_dim[self.frame, :3])
        return np.round(dist / max_len) * max_len

    def azimuth(self, ref_atom):
        dx = self.nn_coords(ref_atom)[:, 0] - self.trj[self.frame, self.probe_vol[ref_atom], 0]
        dy = self.nn_coords(ref_atom)[:, 1] - self.trj[self.frame, self.probe_vol[ref_atom], 1]
        dx -= self.pbc(dx)
        dy -= self.pbc(dy)
        return np.arctan2(dy, dx) + np.pi

    def planar(self, ref_atom):
        dz = self.nn_coords(ref_atom)[:, 2] - self.trj[self.frame, self.probe_vol[ref_atom], 2]
        dz -= self.pbc(dz)
        _, nn_dist = self.nn()
        return np.arccos(dz / nn_dist[ref_atom])

    def q_lm(self, l, ref_atom):
        N_nn = self.nn_coords(ref_atom).shape[0]
        q_lm = []
        for m in range(-l, l + 1):
            q_lm.append(sph_harm(m, l, self.azimuth(ref_atom), self.planar(ref_atom)).sum() / N_nn)
        return np.array(q_lm)

    def Q_lm(self, l):
        N_v = len(self.probe_vol)
        Q_l = np.zeros(2 * l + 1, dtype=np.complex128)
        for i in range(N_v):
            Q_l += self.q_lm(l, i)
        return Q_l / N_v

    def Q_l(self, l):
        Q_lm = self.Q_lm(l)
        prefactor = np.sqrt((4.0 * np.pi) / (2.0 * l + 1))
        Q_l = prefactor * np.sqrt((np.abs(Q_lm) ** 2).sum())
        return Q_l
