#http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html

import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

# Number of samples per component
n_samples = 400

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
#X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
#          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
#
#print X

y0 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=100)
y1 = np.random.multivariate_normal([10, 10], [[1, 0], [0, 1]], size=100)
y2 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], size=100)
y3 = np.random.multivariate_normal([-2, 8], [[1, 0], [0, 1]], size=100)
y = np.vstack([y0, y1, y2, y3])

#print y
## Fit a mixture of Gaussians with EM using five components
gmm = mixture.GMM(n_components=4, covariance_type='full')
gmm.fit(y)
#
## Fit a Dirichlet process mixture of Gaussians using five components
dpgmm = mixture.DPGMM(n_components=5, covariance_type='full')
dpgmm.fit(y)
#
color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
#
for i, (clf, title) in enumerate([(gmm, 'GMM'),
                                  (dpgmm, 'Dirichlet Process GMM')]):
    #i = 1
    splot = plt.subplot(2, 1, 1 + i)
    Y_ = clf.predict(y) #predicts cluster label

    for i, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
    # as the DP will not use every component it has access to
#        # unless it needs it, we shouldn't plot the redundant
#        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(y[Y_ == i, 0], y[Y_ == i, 1], .8, color=color)
        print y[Y_ == i, 0]
        print i
#   
#        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
#
    m1 = max(y[:,0])
    m2 = max(y[:,1])
    plt.xlim(min(y[:,0]), m1)
    plt.ylim(min(y[:,1]), m2)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
#
plt.show()