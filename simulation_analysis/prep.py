"""
@author Kevin Juan
Analyze GROMACS trajectory files for water/ice
Dependencies:
- MDAnalysis
- numpy
"""

from MDAnalysis import*
import numpy as np


class trj_prep:
    def __init__(self, xtc, gro):
        self.xtc = xtc
        self.gro = gro
        self.dr = 0.005
        self.u = Universe(self.gro, self.xtc)

    def species(self):
        i = 0
        while self.u.atoms.resids[i] == 1:
            i += 1
        return list(self.u.atoms[:i])

    def frames(self):
        frames = self.u.trajectory
        return frames, len(frames)

    def select_traj(self, selection_type, frames, n_frames, sample_size):
        selection = self.u.select_atoms(selection_type)
        traj = np.zeros((int(n_frames * sample_size), len(selection), 3), dtype=np.float32)
        n_atoms = len(selection)
        i = 0
        for ts in frames:
            if ts.frame > int(n_frames * (1 - sample_size)):
                traj[i, :, :3] = selection.positions / 10
                i += 1
        n_samples = int(n_frames * sample_size)
        return traj, n_atoms, n_samples

    def box_size(self, frames, n_frames, sample_size):
        box_dim = np.zeros((int(n_frames * sample_size), 6), dtype=np.float32)
        i = 0
        for ts in frames:
            if ts.frame > int(n_frames * (1 - sample_size)):
                box_dim[i] = np.r_[frames.ts.dimensions[0:3] / 10, frames.ts.dimensions[3:]]
                i += 1
        return box_dim

    def bins(self, box_dim):
        max_box = np.max(box_dim[:, :3])
        r_vals = np.arange(0, max_box / 2, self.dr, dtype=np.float32)
        v_shell = np.zeros(len(r_vals), dtype=np.float32)
        v_shell[0] = 4.0 / 3.0 * np.pi * self.dr ** 3
        for i in range(1, len(r_vals)):
            v_shell[i] = 4.0 * np.pi * r_vals[i] ** 2 * self.dr
        return v_shell, r_vals
