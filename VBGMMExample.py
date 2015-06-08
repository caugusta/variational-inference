import sklearn
import numpy as np
import matplotlib.pyplot as plt

#from sklearn import mixture

#Generate random observations with two modes centred on 0 and 10, respectively, to use for training
#http://scikit-learn.org/0.5/modules/gmm.html
#np.random.seed(0)
obs = np.concatenate((np.random.randn(100,1), 10 + np.random.randn(100,1)))


#Plot the observations to show the clusters
#y = np.arange(-10., 190., 1)
#plt.plot(obs, y, 'ro')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Data for a two-component Gaussian mixture model')
#plt.show()

#Create a Variational Bayes Gaussian Mixture Model (VBGMM) object with 2 components, everything else set to default values.
#For more information, see http://scikit-learn.org/stable/modules/generated/sklearn.mixture.VBGMM.html#sklearn.mixture.VBGMM.fit

g = sklearn.mixture.VBGMM(n_components=2)

#Estimate model parameters with the variational algorithm
f1 = g.fit(obs)

print f1

#print 'The'



#print s1.alpha

