#http://www.mit.edu/~mattjj/released-code/hsmm/stats_util.py
#THIS ONE
from __future__ import division
import numpy as np
from pypr.clustering import *
from numpy.random import random
from numpy import newaxis as na
import scipy.stats as stats
import scipy.linalg

### Sampling functions

def sample_discrete(dist,size=[]):
    assert (dist >=0).all()
    cumvals = np.cumsum(dist)
    return np.sum(random(size)[...,na] * cumvals[-1] > cumvals, axis=-1)

def sample_niw(mu_0,lmbda_0,kappa_0,nu_0):
    '''
    Returns a sample from the normal/inverse-wishart distribution, conjugate
    prior for (simultaneously) unknown mean and unknown covariance in a
    Gaussian likelihood model. Returns covariance.  '''
    # this is completely copied from Matlab's implementation, ignoring
    # the copyright. I'm sorry.
    # reference: p. 87 in Gelman's Bayesian Data Analysis

    # first sample Sigma ~ IW(lmbda_0^-1,nu_0)
    lmbda = sample_invwishart(lmbda_0,nu_0) # lmbda = np.linalg.inv(sample_wishart(np.linalg.inv(lmbda_0),nu_0))
    # then sample mu | Lambda ~ N(mu_0, Lambda/kappa_0)
    mu = np.random.multivariate_normal(mu_0,lmbda / kappa_0)

    return mu, lmbda

def sample_invwishart(lmbda,dof):
    # TODO make a version that returns the cholesky
    # TODO allow passing in chol/cholinv of matrix parameter lmbda
    n = lmbda.shape[0]
    chol = np.linalg.cholesky(lmbda)

    if (dof <= 81+n) and (dof == np.round(dof)):
        x = np.random.randn(dof,n)
    else:
        x = np.diag(np.sqrt(stats.chi2.rvs(dof-(np.arange(n)))))
        x[np.triu_indices_from(x,1)] = np.random.randn(n*(n-1)/2)
    R = np.linalg.qr(x,'r')
    T = scipy.linalg.solve_triangular(R.T,chol.T).T
    return np.dot(T,T.T)

def sample_wishart(sigma, dof):
    '''
    Returns a sample from the Wishart distn, conjugate prior for precision matrices.
    '''

    n = sigma.shape[0]
    chol = np.linalg.cholesky(sigma)

    # use matlab's heuristic for choosing between the two different sampling schemes
    if (dof <= 81+n) and (dof == round(dof)):
        # direct
        X = np.dot(chol,np.random.normal(size=(n,dof)))
    else:
        A = np.diag(np.sqrt(np.random.chisquare(dof - np.arange(0,n),size=n)))
        A[np.tri(n,k=-1,dtype=bool)] = np.random.normal(size=(n*(n-1)/2.))
        X = np.dot(chol,A)

    return np.dot(X,X.T)
    
#######################
#######################
#######################
#http://www.bayespy.org/_modules/bayespy/inference/vmp/nodes/mixture.html
######################################################################
# Copyright (C) 2011,2012,2014 Jaakko Luttinen
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
######################################################################

######################################################################
# This file is part of BayesPy.
#
# BayesPy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# BayesPy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BayesPy.  If not, see <http://www.gnu.org/licenses/>.
######################################################################

#"""
#Module for the mixture distribution node.
#"""
#
#import warnings
#import numpy as np
#
#from bayespy.utils import misc
#
#from .expfamily import ExponentialFamily, \
#                       ExponentialFamilyDistribution, \
#                       useconstructor
#                       
#from .categorical import Categorical, \
#                         CategoricalMoments
#
#class MixtureDistribution(ExponentialFamilyDistribution):
#    """
#    Class for the VMP formulas of mixture variables.
#    """
#    
#
#    def __init__(self, distribution, cluster_plate, n_clusters, ndims, 
#                 ndims_parents):
#        """
#        Create VMP formula node for a mixture variable
#        """
#        self.distribution = distribution
#        self.cluster_plate = cluster_plate
#        self.ndims = ndims
#        self.ndims_parents = ndims_parents
#        self.K = n_clusters
#
#
#    def compute_message_to_parent(self, parent, index, u, *u_parents):
#        """
#        Compute the message to a parent node.
#        """
#
#        if index == 0:
#
#            # Shape(phi)    = [Nn,..,K,..,N0,Dd,..,D0]
#            # Shape(L)      = [Nn,..,K,..,N0]
#            # Shape(u)      = [Nn,..,N0,Dd,..,D0]
#            # Shape(result) = [Nn,..,N0,K]
#
#            # Compute g:
#            # Shape(g)      = [Nn,..,K,..,N0]
#            g = self.distribution.compute_cgf_from_parents(*(u_parents[1:]))
#            # Reshape(g):
#            # Shape(g)      = [Nn,..,N0,K]
#            if np.ndim(g) < abs(self.cluster_plate):
#                # Not enough axes, just add the cluster plate axis
#                g = np.expand_dims(g, -1)
#            else:
#                # Move the cluster plate axis
#                g = misc.moveaxis(g, self.cluster_plate, -1)
#
#            # Compute phi:
#            # Shape(phi)    = [Nn,..,K,..,N0,Dd,..,D0]
#            phi = self.distribution.compute_phi_from_parents(*(u_parents[1:]))
#            # Move phi axis:
#            # Shape(phi)    = [Nn,..,N0,K,Dd,..,D0]
#            for ind in range(len(phi)):
#                if self.cluster_plate < 0:
#                    axis_from = self.cluster_plate-self.ndims[ind]
#                else:
#                    raise RuntimeError("Cluster plate axis must be negative")
#                axis_to = -1-self.ndims[ind]
#                if np.ndim(phi[ind]) >= abs(axis_from):
#                    # Cluster plate axis exists, move it to the correct position
#                    phi[ind] = misc.moveaxis(phi[ind], axis_from, axis_to)
#                else:
#                    # No cluster plate axis, just add a new axis to the correct
#                    # position, if phi has something on that axis
#                    if np.ndim(phi[ind]) >= abs(axis_to):
#                        phi[ind] = np.expand_dims(phi[ind], axis=axis_to)
#
#            # Reshape u:
#            # Shape(u)      = [Nn,..,N0,1,Dd,..,D0]
#            u_self = list()
#            for ind in range(len(u)):
#                u_self.append(np.expand_dims(u[ind],
#                                             axis=(-1-self.ndims[ind])))
#
#            # Compute logpdf:
#            # Shape(L)      = [Nn,..,N0,K]
#            L = self.distribution.compute_logpdf(u_self, phi, g, 0, self.ndims)
#
#            # Sum over other than the cluster dimensions? No!
#            # Hmm.. I think the message passing method will do
#            # that automatically
#
#            m = [L]
#
#            return m
#
#        elif index >= 1:
#
#            # Parent index for the distribution used for the
#            # mixture.
#            index = index - 1
#
#            # Reshape u:
#            # Shape(u)      = [Nn,..1,..,N0,Dd,..,D0]
#            u_self = list()
#            for ind in range(len(u)):
#                if self.cluster_plate < 0:
#                    cluster_axis = self.cluster_plate - self.ndims[ind]
#                else:
#                    cluster_axis = self.cluster_plate
#                u_self.append(np.expand_dims(u[ind], axis=cluster_axis))
#
#            # Message from the mixed distribution
#            m = self.distribution.compute_message_to_parent(parent,
#                                                            index, 
#                                                            u_self, 
#                                                            *(u_parents[1:]))
#
#            # Weigh the messages with the responsibilities
#            for i in range(len(m)):
#
#                # Shape(m)      = [Nn,..,K,..,N0,Dd,..,D0]
#                # Shape(p)      = [Nn,..,N0,K]
#                # Shape(result) = [Nn,..,K,..,N0,Dd,..,D0]
#
#                # Number of axes for the variable dimensions for
#                # the parent message.
#                D = self.ndims_parents[index][i]
#
#                # Responsibilities for clusters are the first
#                # parent's first moment:
#                # Shape(p)      = [Nn,..,N0,K]
#                p = u_parents[0][0]
#                # Move the cluster axis to the proper place:
#                # Shape(p)      = [Nn,..,K,..,N0]
#                p = misc.atleast_nd(p, abs(self.cluster_plate))
#                p = misc.moveaxis(p, -1, self.cluster_plate)
#                # Add axes for variable dimensions to the contributions
#                # Shape(p)      = [Nn,..,K,..,N0,1,..,1]
#                p = misc.add_trailing_axes(p, D)
#
#                if self.cluster_plate < 0:
#                    # Add the variable dimensions
#                    cluster_axis = self.cluster_plate - D
#
#                # Add axis for clusters:
#                # Shape(m)      = [Nn,..,1,..,N0,Dd,..,D0]
#                #m[i] = np.expand_dims(m[i], axis=cluster_axis)
#
#                #
#                # TODO: You could do summing here already so that
#                # you wouldn't compute huge matrices as
#                # intermediate result. Use einsum.
#
#                # Compute the message contributions for each
#                # cluster:
#                # Shape(result) = [Nn,..,K,..,N0,Dd,..,D0]
#                m[i] = m[i] * p
#
#            return m
#
#        
#    def compute_mask_to_parent(self, index, mask):
#        """
#        Maps the mask to the plates of a parent.
#        """
#        if index == 0:
#            return mask
#        else:
#            if self.cluster_plate >= 0:
#                raise ValueError("Cluster plate axis must be negative")
#            if np.ndim(mask) >= abs(self.cluster_plate):
#                mask = np.expand_dims(mask, axis=self.cluster_plate)
#            return self.distribution.compute_mask_to_parent(index-1, mask)
#
#        
#    def compute_phi_from_parents(self, mask=True, *u_parents):
#        """
#        Compute the natural parameter vector given parent moments.
#        """
#        # Compute weighted average of the parameters
#
#        # Cluster parameters
#        Phi = self.distribution.compute_phi_from_parents(*(u_parents[1:]))
#        # Contributions/weights/probabilities
#        P = u_parents[0][0]
#
#        phi = list()
#
#        nans = False
#
#        for ind in range(len(Phi)):
#            # Compute element-wise product and then sum over K clusters.
#            # Note that the dimensions aren't perfectly aligned because
#            # the cluster dimension (K) may be arbitrary for phi, and phi
#            # also has dimensions (Dd,..,D0) of the parameters.
#            # Shape(phi)    = [Nn,..,K,..,N0,Dd,..,D0]
#            # Shape(p)      = [Nn,..,N0,K]
#            # Shape(result) = [Nn,..,N0,Dd,..,D0]
#            # General broadcasting rules apply for Nn,..,N0, that is,
#            # preceding dimensions may be missing or dimension may be
#            # equal to one. Probably, shape(phi) has lots of missing
#            # dimensions and/or dimensions that are one.
#
#            if self.cluster_plate < 0:
#                cluster_axis = self.cluster_plate - self.ndims[ind]
#            else:
#                raise RuntimeError("Cluster plate should be negative")
#
#            # Move cluster axis to the last:
#            # Shape(phi)    = [Nn,..,N0,Dd,..,D0,K]
#            if np.ndim(Phi[ind]) >= abs(cluster_axis):
#                phi.append(misc.moveaxis(Phi[ind], cluster_axis, -1))
#            else:
#                phi.append(Phi[ind][...,None])
#
#            # Add axes to p:
#            # Shape(p)      = [Nn,..,N0,K,1,..,1]
#            p = misc.add_trailing_axes(P, self.ndims[ind])
#            # Move cluster axis to the last:
#            # Shape(p)      = [Nn,..,N0,1,..,1,K]
#            p = misc.moveaxis(p, -(self.ndims[ind]+1), -1)
#
#            # Now the shapes broadcast perfectly and we can sum
#            # p*phi over the last axis:
#            # Shape(result) = [Nn,..,N0,Dd,..,D0]
#            phi[ind] = misc.sum_product(p, phi[ind], axes_to_sum=-1)
#            if np.any(np.isnan(phi[ind])):
#                nans = True
#
#        if nans:
#            warnings.warn("The natural parameters of mixture distribution "
#                          "contain nans. This may happen if you use fixed "
#                          "parameters in your model. Technically, one possible "
#                          "reason is that the cluster assignment probability "
#                          "for some element is zero (p=0) and the natural "
#                          "parameter of that cluster is -inf, thus "
#                          "0*(-inf)=nan. Solution: Use parameters that assign "
#                          "non-zero probabilities for the whole domain.")
#            
#        return phi
#
#    
#    def compute_moments_and_cgf(self, phi, mask=True):
#        """
#        Compute the moments and :math:`g(\phi)`.
#        """
#        return self.distribution.compute_moments_and_cgf(phi, mask=mask)
#
#    
#    def compute_cgf_from_parents(self, *u_parents):
#        """
#        Compute :math:`\mathrm{E}_{q(p)}[g(p)]`
#        """
#
#        # Compute weighted average of g over the clusters.
#
#        # Shape(g)      = [Nn,..,K,..,N0]
#        # Shape(p)      = [Nn,..,N0,K]
#        # Shape(result) = [Nn,..,N0]
#
#        # Compute g for clusters:
#        # Shape(g)      = [Nn,..,K,..,N0]
#        g = self.distribution.compute_cgf_from_parents(*(u_parents[1:]))
#
#        # Move cluster axis to last:
#        # Shape(g)      = [Nn,..,N0,K]
#        if np.ndim(g) < abs(self.cluster_plate):
#            # Not enough axes, just add the cluster plate axis
#            g = np.expand_dims(g, -1)
#        else:
#            # Move the cluster plate axis
#            g = misc.moveaxis(g, self.cluster_plate, -1)
#
#        # Cluster assignments/contributions/probabilities/weights:
#        # Shape(p)      = [Nn,..,N0,K]
#        p = u_parents[0][0]
#
#        # Weighted average of g over the clusters. As p and g are
#        # properly aligned, you can just sum p*g over the last
#        # axis and utilize broadcasting:
#        # Shape(result) = [Nn,..,N0]
#
#        g = misc.sum_product(p, g, axes_to_sum=-1)
#
#        return g
#
#    
#    def compute_fixed_moments_and_f(self, x, mask=True):
#        """
#        Compute the moments and :math:`f(x)` for a fixed value.
#        """
#        return self.distribution.compute_fixed_moments_and_f(x, mask=True)
#
#    
#    def plates_to_parent(self, index, plates):
#        """
#        Resolves the plate mapping to a parent.
#
#        Given the plates of the node's moments, this method returns the plates
#        that the message to a parent has for the parent's distribution.
#        """
#        if index == 0:
#            return plates
#        else:
#
#            # Add the cluster plate axis
#            plates = list(plates)
#            if self.cluster_plate < 0:
#                knd = len(plates) + self.cluster_plate + 1
#            else:
#                raise RuntimeError("Cluster plate axis must be negative")
#            plates.insert(knd, self.K)
#            plates = tuple(plates)
#
#            return self.distribution.plates_to_parent(index-1, plates)
#
#        
#    def plates_from_parent(self, index, plates):
#        """
#        Resolve the plate mapping from a parent.
#
#        Given the plates of a parent's moments, this method returns the plates
#        that the moments has for this distribution.
#        """
#        if index == 0:
#            return plates
#        else:
#            plates = self.distribution.plates_from_parent(index-1, plates)
#            
#            # Remove the cluster plate, if the parent has it
#            plates = list(plates)
#            if len(plates) >= abs(self.cluster_plate):
#                plates.pop(self.cluster_plate)
#            return tuple(plates)
#
#
#    def random(self, plates=None, *phi):
#        """
#        Draw a random sample from the distribution.
#        """
#        return self.distribution.random(*phi, plates=plates)
#
#
#class Mixture(ExponentialFamily):
#    r"""
#    Node for exponential family mixture variables.
#
#    The node represents a random variable which is sampled from a
#    mixture distribution. It is possible to mix any exponential family
#    distribution. The probability density function is
#
#    .. math::
#
#        p(x|z=k,\boldsymbol{\theta}_0,\ldots,\boldsymbol{\theta}_{K-1})
#        = \phi(x|\boldsymbol{\theta}_k),
#
#    where :math:`\phi` is the probability density function of the mixed
#    exponential family distribution and :math:`\boldsymbol{\theta}_0,
#    \ldots, \boldsymbol{\theta}_{K-1}` are the parameters of each
#    cluster.  For instance, :math:`\phi` could be the Gaussian
#    probability density function :math:`\mathcal{N}` and
#    :math:`\boldsymbol{\theta}_k = \{\boldsymbol{\mu}_k,
#    \mathbf{\Lambda}_k\}` where :math:`\boldsymbol{\mu}_k` and
#    :math:`\mathbf{\Lambda}_k` are the mean vector and precision matrix
#    for cluster :math:`k`.
#
#    Parameters
#    ----------
#
#    z : categorical-like node or array
#        :math:`z`, cluster assignment
#
#    node_class : stochastic exponential family node class
#        Mixed distribution
#
#    params : types specified by the mixed distribution
#    
#        Parameters of the mixed distribution.  If some parameters should
#        vary between clusters, those parameters' plate axis
#        `cluster_plate` should have a size which equals the number of
#        clusters. For parameters with shared values, that plate axis
#        should have length 1. At least one parameter should vary between
#        clusters.
#
#    cluster_plate : int, optional
#    
#        Negative integer defining which plate axis is used for the
#        clusters in the parameters. That plate axis is ignored from the
#        parameters when considering the plates for this node. By
#        default, mix over the last plate axis.
#
#    See also
#    --------
#
#    Categorical, CategoricalMarkovChain
#
#    Examples
#    --------
#
#    A simple 2-dimensional Gaussian mixture model with three clusters
#    for 100 samples can be constructed, for instance, as:
#
#    .. code-block:: python
#
#        from bayespy.nodes import (Dirichlet, Categorical, Mixture,
#                                   Gaussian, Wishart)
#        alpha = Dirichlet([1e-3, 1e-3, 1e-3])
#        Z = Categorical(alpha, plates=(100,))
#        mu = Gaussian(np.zeros(2), 1e-6*np.identity(2), plates=(3,))
#        Lambda = Wishart(2, 1e-6*np.identity(2), plates=(3,))
#        X = Mixture(Z, Gaussian, mu, Lambda)
#    """
#
#
#    def __init__(self, z, node_class, cluster_plate=-1, *params, **kwargs):
#        self.cluster_plate = cluster_plate
#        super().__init__(z, node_class, *params, cluster_plate=cluster_plate,
#                         **kwargs)
#        
#
#    @classmethod
#    def _constructor(cls, z, node_class, cluster_plate=-1, *args, **kwargs):
#        """
#        Constructs distribution and moments objects.
#        """
#        if cluster_plate >= 0:
#            raise ValueError("Cluster plate axis must be negative")
#        
#        # Get the stuff for the mixed distribution
#        (parents, _, dims, mixture_plates, distribution, moments, parent_moments) = \
#          node_class._constructor(*args)
#
#        # Check that at least one of the parents has the cluster plate axis
#        if len(mixture_plates) < abs(cluster_plate):
#            raise ValueError("The mixed distribution does not have a plates "
#                             "axis for the cluster plate axis")
#        
#        # Resolve the number of clusters
#        mixture_plates = list(mixture_plates)
#        K = mixture_plates.pop(cluster_plate)
#        
#        # Convert a node to get the number of clusters
#        z = cls._ensure_moments(z, CategoricalMoments(K))
#        if z.dims[0][0] != K:
#            raise ValueError("Inconsistent number of clusters")
#
#        plates = cls._total_plates(kwargs.get('plates'), mixture_plates, z.plates)
#
#        ndims = [len(dim) for dim in dims]
#        parents = [cls._ensure_moments(p_i, m_i) 
#                   for (p_i, m_i) in zip(parents, parent_moments)]
#        ndims_parents = [[len(dims_i) for dims_i in parent.dims]
#                         for parent in parents]
#                          
#        
#        # Convert the distribution to a mixture
#        distribution = MixtureDistribution(distribution, 
#                                           cluster_plate,
#                                           K,
#                                           ndims,
#                                           ndims_parents)
#
#        # Add cluster assignments to parents
#        parent_moments = [CategoricalMoments(K)] + list(parent_moments)
#
#        parents = [z] + list(parents)
#
#        return (parents,
#                kwargs,
#                dims,
#                plates,
#                distribution, 
#                moments, 
#                parent_moments)
#    
#
#    def integrated_logpdf_from_parents(self, x, index):
#
#        """ Approximates the posterior predictive pdf \int p(x|parents)
#        q(parents) dparents in log-scale as \int q(parents_i) exp( \int
#        q(parents_\i) \log p(x|parents) dparents_\i ) dparents_i."""
#
#        if index == 0:
#            # Integrate out the cluster assignments
#
#            # First, integrate the cluster parameters in log-scale
#
#            # compute_logpdf(cls, u, phi, g, f):
#
#            # Shape(x) = [M1,..,Mm,N1,..,Nn,D1,..,Dd]
#
#            u_parents = self._message_from_parents()
#
#            # Shape(u) = [M1,..,Mm,N1,..,1,..,Nn,D1,..,Dd]
#            # Shape(f) = [M1,..,Mm,N1,..,1,..,Nn]
#            (u, f) = self._distribution.distribution.compute_fixed_moments_and_f(x)
#            f = np.expand_dims(f, axis=self.cluster_plate)
#            for i in range(len(u)):
#                ndim_i = len(self.dims[i])
#                cluster_axis = self.cluster_plate - ndim_i
#                u[i] = np.expand_dims(u[i], axis=cluster_axis)
#            # Shape(phi) = [N1,..,K,..,Nn,D1,..,Dd]
#            phi = self._distribution.distribution.compute_phi_from_parents(*(u_parents[1:]))
#            # Shape(g) = [N1,..,K,..,Nn]
#            g = self._distribution.distribution.compute_cgf_from_parents(*(u_parents[1:]))
#            # Shape(lpdf) = [M1,..,Mm,N1,..,K,..,Nn]
#            lpdf = self._distribution.distribution.compute_logpdf(u, phi, g, f, self.ndims)
#
#            # From logpdf to pdf, but avoid over/underflow
#            lpdf_max = np.max(lpdf, axis=self.cluster_plate, keepdims=True)
#            pdf = np.exp(lpdf-lpdf_max)
#
#            # Move cluster axis to be the last:
#            # Shape(pdf) = [M1,..,Mm,N1,..,Nn,K]
#            pdf = misc.moveaxis(pdf, self.cluster_plate, -1)
#
#            # Cluster assignments/probabilities/weights
#            # Shape(p) = [N1,..,Nn,K]
#            p = u_parents[0][0]
#
#            # Weighted average. TODO/FIXME: Use einsum!
#            # Shape(pdf) = [M1,..,Mm,N1,..,Nn]
#            pdf = np.sum(pdf * p, axis=self.cluster_plate)
#
#            # Back to log-scale (add the overflow fix!)
#            lpdf_max = np.squeeze(lpdf_max, axis=self.cluster_plate)
#            lpdf = np.log(pdf) + lpdf_max
#
#            return lpdf
#
#        raise NotImplementedError()    
#    
########################
########################
########################    
#    
##http://www.bayespy.org/_modules/bayespy/plot.html#gaussian_mixture_2d
#    
#def gaussian_mixture_2d(X, alpha=None, scale=2, fill=False, axes=None, **kwargs):
#    """
#    Plot Gaussian mixture as ellipses in 2-D
#
#    Parameters
#    ----------
#
#    X : Mixture node
#
#    alpha : Dirichlet-like node (optional)
#       Probabilities for the clusters
#
#    scale : float (optional)
#       Scale for the covariance ellipses (by default, 2)
#    """
#        
#    if axes is None:
#        axes = plt.gca()
#
#    mu_Lambda = X.parents[1]._convert(GaussianWishartMoments)
#
#    (mu, _, Lambda, _) = mu_Lambda.get_moments()
#    mu = np.linalg.solve(Lambda, mu)
#
#    if len(mu_Lambda.plates) != 1:
#        raise NotImplementedError("Not yet implemented for more plates")
#    
#    K = mu_Lambda.plates[0]
#
#    width = np.zeros(K)
#    height = np.zeros(K)
#    angle = np.zeros(K)
#
#    for k in range(K):
#        m = mu[k]
#        L = Lambda[k]
#        (u, W) = scipy.linalg.eigh(L)
#        u[0] = np.sqrt(1/u[0])
#        u[1] = np.sqrt(1/u[1])
#        width[k] = 2*u[0]
#        height[k] = 2*u[1]
#        angle[k] = np.arctan(W[0,1] / W[0,0])
#
#    angle = 180 * angle / np.pi
#    mode_height = 1 / (width * height)
#
#    # Use cluster probabilities to adjust alpha channel
#    if alpha is not None:
#        # Compute the normalized probabilities in a numerically stable way
#        logsum_p = misc.logsumexp(alpha.u[0], axis=-1, keepdims=True)
#        logp = alpha.u[0] - logsum_p
#        p = np.exp(logp)
#        # Visibility is based on cluster mode peak height
#        visibility = mode_height * p
#        visibility /= np.amax(visibility)
#    else:
#        visibility = np.ones(K)
#
#    for k in range(K):
#        ell = mpl.patches.Ellipse(mu[k], scale*width[k], scale*height[k],
#                                  angle=(180+angle[k]),
#                                  fill=fill,
#                                  alpha=visibility[k],
#                                  **kwargs)
#        axes.add_artist(ell)
#
#    plt.axis('equal')
#
#    # If observed, plot the data too
#    if np.any(X.observed):
#        mask = np.array(X.observed) * np.ones(X.plates, dtype=np.bool)
#        y = X.u[0][mask]
#        plt.plot(y[:,0], y[:,1], 'r.')
#
#    return
    
#First generate data in 4 clusters, with 100 individuals per cluster
#For example, y0 has mean [0,0] and identity covariance matrix
y0 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=100)
y1 = np.random.multivariate_normal([10, 10], [[1, 0], [0, 1]], size=100)
y2 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], size=100)
y3 = np.random.multivariate_normal([-2, 8], [[1, 0], [0, 1]], size=100)
y = np.vstack([y0, y1, y2, y3])
#y = y0

N = 400 #The number of data points
D = 2 #we're in 2 dimensions
K = 4

mu= np.random.multivariate_normal(np.zeros(D), 1e-5*np.identity(D), K)

Lambda1 = sample_wishart(1e-5*np.identity(D), D)
Lambda2 = sample_wishart(1e-5*np.identity(D), D)
Lambda3 = sample_wishart(1e-5*np.identity(D), D)
Lambda4 = sample_wishart(1e-5*np.identity(D), D)

Lambda = np.vstack([Lambda1, Lambda2, Lambda3, Lambda4])

p1 = gmm.em(y, K)

print p1
