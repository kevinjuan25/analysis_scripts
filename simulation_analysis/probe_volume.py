"""
@author Kevin Juan
Orientational order parameters
Dependencies:
- MDAnalysis
- numpy
"""

from MDAnalysis import*
from MDAnalysis import lib
import numpy as np


class probe_volume:
    def __init__(self, trj, frame, box_dim, radius, center):
        self.trj = trj
        self.frame = frame
        self.box_dim = box_dim
        self.radius = radius
        self.center = center

    def probe_center(self):
        center = np.zeros((self.trj.shape[1], 3))
        center[:, 0] = self.center[0]
        center[:, 1] = self.center[1]
        center[:, 2] = self.center[2]
        return center

    def probe_sph(self):
        probe_center = self.probe_center()

        if np.any(probe_center[0] > self.box_dim[self.frame, :3]):
            raise Exception("Probe center lies outside the box!")

        vol_search = lib.nsgrid.FastNS(self.radius, self.trj[self.frame], self.box_dim[self.frame], pbc=True)
        probe_idx = np.array(vol_search.search(probe_center).get_indices()[0])
        sorted_idx = np.sort(probe_idx)
        return sorted_idx
