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

    def pbc(self, dist):
        max_len = max(self.box_dim[self.frame, :3])
        return np.round(dist / max_len) * max_len

    def angles_coords(self, ref_atom):
        nn_idx, nn_dist = self.nn()
        coords = self.trj[self.frame, nn_idx[ref_atom]]
        dr = coords - self.trj[self.frame, self.probe_vol[ref_atom]]
        dr -= self.pbc(dr)
        azimuth = np.arctan2(dr[:, 1], dr[:, 0]) + np.pi
        planar = np.arccos(dr[:, 2] / nn_dist[ref_atom])
        return azimuth, planar, coords

    def q_lm(self, l, ref_atom):
        azimuth, planar, nn_coords = self.angles_coords(ref_atom)
        N_nn = len(nn_coords)
        q_lm = []
        for m in range(-l, l + 1):
            q_lm.append(sph_harm(m, l, azimuth, planar).sum() / N_nn)
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
