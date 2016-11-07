from shapesorting import *
from sandbox.util import Saver

import matplotlib.pyplot as plt

import argparse
import numpy as np
import sklearn as sk
import sklearn.metrics as mets

from sklearn.decomposition import *
from sklearn.lda import LDA

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_time',type=str,default='16-11-05')
args = parser.parse_args()

encoding_saver = Saver(time=args.encoding_time,path='{}/{}'.format(DATADIR,'enco_simi_data'))

Y = encoding_saver.load_value(0,'Y')
N = len(Y)
D = encoding_saver.load_dictionary(0, 'l4_encodings')
C = np.array([[1,0,0],
              [0,0,1]])

pca = PCA(n_components=2)
#lda = LDA(n_components=2,solver='svd')
#lda = sk.discriminant_analysis.LinearDiscriminantAnalysis(n_components=5)

for merge in [lambda x, y : np.column_stack((x,y)), lambda x, y : x * y]:
    
    X = merge(D['bZ1'],D['bZ2'])
    #lda.fit(X, Y)
    #P = lda.means_
    #Z = lda.transform(X)
    #Z = np.dot(X,P.T)
    Z = pca.fit_transform(X, y=Y)
    if Z.shape[-1] == 1:
        plt.scatter(Z[:,0],np.ones(N,), c= C[Y])
        plt.show()
    else:
        plt.scatter(Z[:,0], Z[:,1], c=C[Y])
        plt.show()
        
    halt= True
    
    
    
    #N = X.shape[0]
    #p = np.random.permutation(N)
    
    #X = X[p]
    #Y = Y[p]
    #cutoff = int(0.7 * N)
    
    #X_t = X[:cutoff]
    #Y_t = Y[:cutoff]
    #X_v = X[cutoff:]
    #Y_v = Y[cutoff:]