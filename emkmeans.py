##############################################################################
# This file was (almost) automatically extracted from the TeXmacs document   #
# where it was developed. If actually makes more sense for it to be read in  #
# this manner (interspersed explanations, etc.), but distribution and reuse  #
# are just impossible then. Furthermore, the plotting functions return the   #
# figures, as this is needed for ps_out() inside TeXmacs. Just do pl.show()  #
# if you are not running inside TeXmacs.                                     #
#                                                                            #
# You will need a copy of the old-faithful dataset (provided in the package) #
#                                                                            #
# Copyleft: (C) Miguel de Benito, October 2014                               #
# You may do whatever you like with this file and its contents.              #
# This message will self-destruct in 5 seconds.                              #
##############################################################################

from __future__ import print_function #
from __future__ import division       # Sensible division from Python3
import numpy as np
import numpy.random as ra
import matplotlib as mp
import matplotlib.pyplot as pl
from numpy.linalg import norm
from math import pi

#import logging as log
#log.basicConfig(filename='/tmp/tm_python.log',level=log.DEBUG)


def normalize(x, mu=None, sigma=None):
    """Standardize x values using local or provided average and stddev.
    
    Cases along the first dimension(axis 0),
    dimensions along the 2nd(axis 1).
    """
    x = np.atleast_2d(x)
    
    avg = mu if mu is not None else np.mean(x, axis=0)
    std = sigma if sigma is not None and sigma is not 0\
          else np.std(x, axis=0)
    
    return (x - avg) / std


def get_plot(points, prototypes=None, groups=None, cmap='jet'):
    """Plots an $N \times 2$ array of $N$ points in $\mathbb{R}^2$.

    prototypes= $K \times 2$ array of $K$ points in $\mathbb{R}^2$, each will be 
                drawn with a red cross 
    groups= $N \times 1$ array of assignments point -> group for coloring
    """
    k = len(prototypes) if groups is not None else 1
    n = len(points)
    cNorm = mp.colors.Normalize(vmin=0, vmax=k-1)
    scalarMap = mp.cm.ScalarMappable(norm=cNorm, cmap=pl.get_cmap(cmap))
    colors = np.zeros((n, 4))
    for i in range(n):
        colors[i] = scalarMap.to_rgba(groups[i] if groups is not None else 0)
    
    #pl.clf()
    pl.scatter(points[:,0], points[:,1], marker='o', c=colors, edgecolors='none')
    if prototypes.any():
        pl.scatter(prototypes[:,0], prototypes[:,1], marker='x', c='r', s=100)

    return pl.gcf() # Needed for usage inside TeXmacs 


def gen_data(n=300, k=3, d=2, normed=True):
    data  = np.zeros((n,d))
    means = ra.random((k,d))*10-5    # Hack hack hack...
    covs  = np.empty(k, dtype=np.matrix)
    for i in range(k):
        covs[i] = np.matrix(np.zeros((d,d)))
        while np.linalg.det(covs[i]) == 0:
            S = np.matrix(ra.random((d, d))*2 - 1)
            covs[i] = S * S.T / 2

    rs = ra.RandomState()
    for i in range(n):
        j = int(ra.random()*k)
        data[i] = rs.multivariate_normal(means[j], covs[j])
    if normed:
        return [normalize(means), covs, normalize(data)]
    else:
        return [means, covs, data]


# FIXME: Those loops can almost surely be vectorized...
class KMeans:
    """ KMeans algorithm. See the doc for a description. """
    
    def __init__(self, k, X, maxiter=20):
        self.k = k               # Number of prototypes / clusters
        self.X = X               # Normalized data
        self.n, self.d = X.shape # Number and dimension of data points
        self.MU = None           # The d coordinates of the k prototypes
        self.j = 0.0             # "distortion measure" (quantity to optimize)
        self.R = None            # Assignments: R_n = k iff X_n is assigned to MU_k
        self.maxiter = maxiter
        self.restart()

    def restart(self):
        self.j = 0.0
        # FIXME: Remove hardcoded variables or normalize here
        self.MU = ra.random((self.k, self.d)) * 4 - 2
        self.R = ra.random(size=self.n) * self.k
        self.R = self.R.astype(np.int8)

    def update_distortion(self):
        self.j = 0.0
        for n, x in enumerate(self.X):
            self.j += norm(x - self.MU[self.R[n]], ord=2)**2
        
    def minimize_assignments(self):
        """ Minimize R_nk for fixed MU_k. """
        changes = False
        for n, x in enumerate(self.X):
            # FIXME: Reuse results from previous computations
            cur = norm(x - self.MU[self.R[n]], ord=2)**2
            for k, m in enumerate(self.MU):
                val = norm(x - m, ord=2)**2
                if val < cur:
                    changes = True
                    self.R[n] = k
        return changes

    def minimize_prototypes(self):
        """ Minimize MU_k for fixed R_nk. """
        for k in range(self.k): 
            idx = self.R == k
            try:
                self.MU[k] = np.sum(self.X[idx], axis=0) / np.count_nonzero(idx)
            except:  # Sloppy (use np.seterr()?)
                continue
        
    def run(self):
        J = []
        for step in range(self.maxiter):
            #print("step= {}, j= {}".format(step, self.j))
            changes = self.minimize_assignments()
            self.update_distortion()  # Store for the plots
            J.append(self.j)
            
            self.minimize_prototypes()
            self.update_distortion()
            J.append(self.j)
            
            if not changes: 
                break

        print("Iterations = {}, Distortion = {}".format(step, self.j))
        return J


# There is certainly some module in numpy or scipy with a function to
# evaluate a Gaussian density in R^n but we can also implement our own
# inefficient hack:

# We'd rather use some sort of static variable inside the function...
mnormal_c = 1.0 / np.sqrt(2*pi)
def mnormal(X, M, S, Sdri=None, SI=None):
    """Evaluate density of multivariate normal.
    X=point, M=mean, S=covariance matrix
    Sdri=inverse of square root of determinant of S
    SI= inverse of S
    """
    if Sdri is None:
        Sdri = 1.0 / np.sqrt(np.linalg.det(S))
    if SI is None:
        SI = S.getI()
    c = mnormal_c * Sdri
    v = X - M
    ex = -0.5 * np.einsum('ai,ij,bj', v, SI, v)
    return float(c * np.exp(ex))


class EMGauss:
    """Sloppy implementation of EM for Gaussian mixtures.

    FIXME: This is sloooooooooow...
    FIXME: Will throw divide by zero if $\mu_k = x_n$ for some $k, n$!
    """
    def __init__(self, X, k=2, maxiter=10):
        self.k = k               # Number of Gaussians
        self.X = X               # Normalized data
        self.n, self.d = X.shape # Number and dimension of data points
        self.maxiter = maxiter
        self.restart()

    def restart(self):
        # Means of the Gaussians
        # FIXME: Remove hardcoded variables or normalize here
        self.M = np.matrix(ra.random((self.k, self.d))*4 - 2)
        
        # Covariance matrices of the Gaussians(init to spherical cov.)
        # We store $S^{- 1}, \det(S)^{- 1 / 2}$ for faster evaluation of the density
        # of the multivariate normal.
        self.S    = np.empty(self.k, dtype = np.matrix)
        self.SI   = np.empty(self.k, dtype = np.matrix)
        self.Sdri = np.empty(self.k, dtype = float)

        for k in range(self.k):
            self.S[k] = np.matrix(np.zeros((self.d, self.d)))
            while np.linalg.det(self.S[k]) == 0:
                self.S[k] = np.matrix(np.diag(ra.random(self.d)))
            self.SI[k] = self.S[k].getI()
            self.Sdri[k] = 1.0 / np.sqrt(np.linalg.det(self.S[k]))

        # Mixing coefficients
        self.P = np.matrix(ra.random((self.k, 1)))
        self.P /= np.sum(self.P)

        # Responsibilities
        self.G = np.matrix(np.zeros((self.n, self.k)))
        self.R = np.matrix(np.zeros((self.k, 1)))

        self.update_responsibilities()
        self.update_loglikelihood()

    def update_responsibilities(self):
        #print("Updating responsibilities")
        for n, x in enumerate(self.X):
            for k in range(self.k):
                self.G[n,k] = self.P[k,0] * mnormal(x, self.M[k], self.S[k],
                                                    Sdri=self.Sdri[k], SI=self.SI[k])
            self.G[n] /= np.sum(self.G[n])
        self.R = np.sum(self.G, axis=0).T

    def update_means(self):
        #print("Updating means")
        for k in range(self.k):
            self.M[k] = np.zeros(self.d)
            for n, x in enumerate(self.X):
                self.M[k] += self.G[n,k] * x
        self.M /= self.R
    
    def update_covariances(self):
        #print("Updating covariances")
        for k in range(self.k):
            self.S[k] = np.matrix(np.zeros((self.d, self.d)))
            for n, x in enumerate(self.X):
                v = np.matrix(x - self.M[k])
                self.S[k] += self.G[n, k] *(v.T * v)
            self.S[k] /= self.R[k]
            self.SI[k] = self.S[k].getI()
            self.Sdri[k] = 1.0 / np.sqrt(np.linalg.det(self.S[k]))

    def update_mixture(self):
        #print("Updating mixture")
        self.P = self.R / self.n

    def update_loglikelihood(self):
        """ Compute
        \[ \log p(\ensuremath{\boldsymbol{x}}|\ensuremath{\boldsymbol{\pi}},
        \ensuremath{\boldsymbol{\mu}}, \ensuremath{\boldsymbol{S}}) = \sum_{n =
        1}^N \log \sum_{k = 1}^K \pi_k \mathcal{N}(x_n | \mu_k, S_k) . \]
        """
        #print("Updating log likelihood")
        # NOTE: that we must repeat this computation because we have
        # new parameter values(cf. update_responsibilities())
        self.ll = 0.0
        for n, x in enumerate(self.X):
            v = 0.0
            for k in range(self.k):
                v += self.P[k,0] * mnormal(x, self.M[k], self.S[k],
                                           Sdri=self.Sdri[k], SI=self.SI[k])
            self.ll += np.log(v)

    def run(self):
        LL = []
        for step in range(self.maxiter):
            self.update_responsibilities()
            self.update_means()
            self.update_covariances()
            self.update_mixture()
            self.update_loglikelihood()
            LL.append(self.ll)
            #print("step= {}, ll= {}".format(step, self.ll))
            if step > 0 and abs(LL[-1] - LL[-2]) < 1e-4: break
        print("Iterations = {}, log likelihood = {}".format(step+1, self.ll))
        return np.array(LL)

    def get_plot(self, cmap='jet'):
        """Plots an $N \times 2$ matrix of $N$ points in $\mathbb{R}^2$.
        ... 
        """
        cNorm = mp.colors.Normalize(vmin = 0, vmax = self.k-1)
        scalarMap = mp.cm.ScalarMappable(norm = cNorm, cmap = pl.get_cmap(cmap))
        colors = np.zeros((self.n, 4))
        for n, p in enumerate(self.G):
            for i in range(self.k):
                rgba = scalarMap.to_rgba(i)
                for c in range(4):
                    colors[n,c] += rgba[c] * p[0,i]
                    colors[n,c] = min(1.0, colors[n,c])
        #pl.clf()
        pl.scatter(self.X[:,0], self.X[:,1], marker='o', c=colors, edgecolors='none')
        pl.scatter(self.M[:,0], self.M[:,1], marker='x', color='r', s=100)
        x = np.linspace(np.min(self.X[:, 0]), np.max(self.X[:, 0]), 400)
        y = np.linspace(np.min(self.X[:, 1]), np.max(self.X[:, 1]), 400)
        xx, yy = np.meshgrid(x, y)
        xy = np.dstack((xx,yy))
        for k in range(self.k):
            zz = np.zeros_like(xx)
            Sdri = 1.0 / np.sqrt(np.linalg.det(self.S[k]))
            SI = self.S[k].getI()
            for i in np.ndindex(xy.shape[:-1]):
                zz[i] = mnormal(xy[i], self.M[k], self.S[k], Sdri, SI)
            pl.contour(xx, yy, zz, alpha=0.5)
        return pl.gcf()

