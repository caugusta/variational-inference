import numpy as np
from numpy.linalg import inv
import sklearn
import itertools

import pylab as pl
import matplotlib as mpl

#http://scikit-learn.org/0.5/auto_examples/gmm/plot_gmm.html#example-gmm-plot-gmm-py

n, m = 300, 2

# generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.7], [3.5, .7]])
#X = np.r_[np.dot(np.random.randn(n, 2), C),
          #np.random.randn(n, 2) + np.array([3, 3])]
          
X1 = np.random.randn(n, 2) + np.array([2,2]) #This has mean (2,2)
X2 = np.random.randn(n, 2) + np.array([10,10]) #this has mean (10,10)

X_use = np.r_[X1, X2]
#yplot = np.arange(0., 600., 1)
pl.plot(X_use[:,0], X_use[:,1], 'ro')
pl.xlabel('x')
pl.ylabel('y')
pl.axis([min(X_use[:,0])-1, max(X_use[:,0])+1, min(X_use[:,1])-1, max(X_use[:,1])+1])
pl.title('Data for a two-component Gaussian mixture model')
pl.show()

clf = sklearn.mixture.gmm.GMM(n_components=2, covariance_type='full')
clf.fit(X_use)

vbex = sklearn.mixture.VBGMM(n_components=2, covariance_type='full')
v1 = vbex.fit(X_use)

print v1.get_params()

#print vbex.means_
#print v1.n_features

splot = pl.subplot(111, aspect='equal')
color_iter = itertools.cycle (['r', 'g', 'b', 'c'])

Y_vbex = vbex.predict(X_use)

Y_ = clf.predict(X_use)

#print vbex.precs_
#vbex_precs = np.array(vbex.precs_)
vbex_covars1 = inv(vbex.precs_)
#print vbex_covars1
#print vbex.means_
#print clf.means_
#print clf.covars_

#for i, (mean, covar, color) in enumerate(zip(vbex.means_, vbex_covars1, color_iter)):
#    v, w = np.linalg.eigh(covar)
#    u = w[0] / np.linalg.norm(w[0])
#    pl.scatter(X_use[Y_vbex==i, 0], X_use[Y_vbex==i, 1], .8, color=color)
#    angle = np.arctan(u[1]/u[0])
#    angle = 180 * angle / np.pi # convert to degrees
#    ell = mpl.patches.Ellipse (mean, v[0], v[1], 180 + angle, color=color)
#    ell.set_clip_box(splot.bbox)
#    ell.set_alpha(0.5)
#    splot.add_artist(ell)
#
#pl.show()

#for i, (mean, covar, color) in enumerate(zip(clf.means_, clf.covars_, color_iter)):
#    v, w = np.linalg.eigh(covar)
#    u = w[0] / np.linalg.norm(w[0])
#    pl.scatter(X_use[Y_==i, 0], X_use[Y_==i, 1], .8, color=color)
#    angle = np.arctan(u[1]/u[0])
#    angle = 180 * angle / np.pi # convert to degrees
#    ell = mpl.patches.Ellipse (mean, v[0], v[1], 180 + angle, color=color)
#    ell.set_clip_box(splot.bbox)
#    ell.set_alpha(0.5)
#    splot.add_artist(ell)
#
#pl.show()