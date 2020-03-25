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
    def __init__(self, trj, frame, box_dim, bound, probe_vol, qBar=False):
        self.trj = trj
        self.frame = frame
        self.box_dim = box_dim
        self.bound = bound
        self.probe_vol = probe_vol
        self.qBar = qBar
        self.n_probe = len(probe_vol)
        self.shell_1_idx = []
        self.shell_1_dist = []

    def nn(self):
        search = lib.nsgrid.FastNS(self.bound, self.trj[self.frame], self.box_dim[self.frame], pbc=True)
        nn_idx = search.search(self.trj[self.frame, self.probe_vol]).get_indices()
        nn_idx = np.array([np.array(x) for x in nn_idx])
        nn_dist = search.search(self.trj[self.frame, self.probe_vol]).get_distances()
        nn_dist = np.array([np.array(x) for x in nn_dist])
        nn_idx, nn_dist = self.nn_clean(nn_idx, nn_dist)

        # Save first coordination shell
        self.shell_1_idx = nn_idx
        self.shell_1_dist = nn_dist
        sys = np.copy(self.probe_vol)

        # Determine NN of probe atoms and NN of probe atoms
        if self.qBar is True:
            for i in range(len(nn_idx)):
                sys = np.append(sys, nn_idx[i])
            sys = np.unique(sys)
            self.new_probe = sys
            nn_idx = search.search(self.trj[self.frame, sys]).get_indices()
            nn_idx = np.array([np.array(x) for x in nn_idx])
            nn_dist = search.search(self.trj[self.frame, sys]).get_distances()
            nn_dist = np.array([np.array(x) for x in nn_dist])
            nn_idx, nn_dist = self.nn_clean(nn_idx, nn_dist, sys)
        return nn_idx, nn_dist, sys

    def nn_clean(self, nn_idx, nn_dist, sys=[]):
        if self.qBar is True:
            self.n_probe = len(sys)
        if len(nn_dist.shape) == 1:
            for i in range(self.n_probe):
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

    def get_shell1(self):
        return self.shell_1_idx, self.shell_1_dist

    def pbc(self, dist):
        max_len = max(self.box_dim[self.frame, :3])
        return np.round(dist / max_len) * max_len

    def angles_coords(self, ref_atom):
        nn_idx, nn_dist, sys = self.nn()
        coords = self.trj[self.frame, nn_idx[ref_atom]]
        if self.qBar is True:
            dr = coords - self.trj[self.frame, sys[ref_atom]]
        else:
            dr = coords - self.trj[self.frame, self.probe_vol[ref_atom]]
        dr -= self.pbc(dr)
        azimuth = np.arctan2(dr[:, 1], dr[:, 0]) + np.pi
        planar = np.arccos(dr[:, 2] / nn_dist[ref_atom])
        return azimuth, planar, coords

    def q_lm(self, l, ref_atom):
        azimuth, planar, nn_coords = self.angles_coords(ref_atom)
        N_nn = len(nn_coords)
        q_lm = np.ndarray(2 * l + 1, dtype=np.complex128)
        for m in range(-l, l + 1):
            q_lm[m + l] = sph_harm(m, l, azimuth, planar).sum() / N_nn
        return q_lm

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

    def qBar_lm(self, l):
        q_i = self.q_lm(l, 0)
        N_sys = self.n_probe
        N_v = len(self.probe_vol)
        qBar_i = np.ndarray((N_v, 2 * l + 1), dtype=np.complex128)
        for i in range(1, N_sys):
            q_i = np.vstack((q_i, self.q_lm(l, i)))

        for i in range(N_v):
            find_nn = np.searchsorted(self.new_probe, self.shell_1_idx[i])
            N_nn = len(self.shell_1_idx[i])
            qBar_i[i] = np.sum(q_i[find_nn], axis=0) / N_nn
        return qBar_i

    def qBar_l(self, l):
        qBar_lm_i = self.qBar_lm(l)
        prefactor = np.sqrt((4.0 * np.pi) / (2.0 * l + 1))
        qBar_l_i = prefactor * np.sqrt(np.sum(np.abs(qBar_lm_i) ** 2, axis=1))
        return qBar_l_i

    def q_l(self, l):
        qBar_l_i = self.qBar_l(l)
        N_v = len(qBar_l_i)
        return qBar_l_i.sum() / N_v
