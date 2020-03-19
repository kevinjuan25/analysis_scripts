"""
@author Kevin Juan
Analyze GROMACS trajectory files for water/ice
Dependencies:
- matplotlib
- scipy
- numpy
"""

from MDAnalysis import*
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import signal
import numpy as np


class post_gr:
    def __init__(self, wd, rdf=0, nr=0, r=0):
        self.rdf = np.average(rdf, axis=0)
        self.nr = np.average(np.cumsum(nr, axis=1), axis=0)
        self.r = r
        self.wd = wd

    def get_gr(self):
        return self.rdf

    def get_nr(self):
        return self.nr

    def peaks(self):
        peak_loc = signal.argrelextrema(self.rdf, np.greater, order=10)
        return peak_loc

    def valleys(self):
        valley_loc = signal.argrelextrema(self.rdf, np.less, order=10)
        return valley_loc

    def plot_gr(self, f_name):
        plt.clf()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(10 * self.r[:300], self.rdf[:300])
        plt.plot([0, 10 * r[300]], [1, 1], ls=':')
        plt.scatter(10 * self.r[self.peaks()[0][:5]], self.rdf[self.peaks()[0][:5]], marker='.', color='red')
        plt.scatter(10 * self.r[self.valleys()[0][:5]], self.rdf[self.valleys()[0][:5]], marker='.', color='green')
        plt.xlabel("$r$" + " in \AA")
        plt.ylabel("$g(r)$")
        plt.savefig(self.wd + '/' + f_name, dpi=300)
        plt.show()
        return

    def plot_nr(self, f_name):
        plt.clf()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(10 * self.r[:300], self.nr[:300])
        plt.xlabel("$r$" + " in \AA")
        plt.ylabel("$N_{ij}(r)$")
        plt.savefig(self.wd + '/' + f_name, dpi=300)
        plt.show()
        return

    def output(self, f_name):
        data = np.c_[self.r[:-1], self.rdf[:-1], self.nr[:-1]]
        np.savetxt(self.wd + '/' + f_name, data, header='r[nm]  g(r)  N(r)')
        return
