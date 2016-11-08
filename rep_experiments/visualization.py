from shapesorting import *
from sandbox.util import Saver

import matplotlib.pyplot as plt

import argparse
import numpy as np
import sklearn as sk
import sklearn.metrics as mets

from sklearn.decomposition import *
from sklearn.lda import LDA
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_time',type=str,default='16-11-06')
parser.add_argument('--use_pca',type=bool,default=False)
parser.add_argument('--use_tsne',type=bool,default=False)
args = parser.parse_args()

encoding_saver = Saver(time=args.encoding_time,path='{}/{}'.format(DATADIR,'enco_simi_data'))

simi_data = encoding_saver.load_dictionary(0,'simi_data')
D = encoding_saver.load_dictionary(0, 'l2_encodings')
C = np.array([[1,0,0],
              [0,0,1]])
K = simi_data['SHAPES1']
Y = simi_data['Y']
P = simi_data['POSITS1']

N = len(Y)

pca = PCA(n_components=2)
tsne = TSNE(n_components=2)

def color_from_2D(X):
    X = X.astype('float32')
    X -= X.min()
    X /= X.max()
    return np.column_stack((X,np.zeros_like(X[:,0])[...,None]))

f, (ax1, ax2) = plt.subplots(1,2)
#K = color_from_2D(P)
for merge in [lambda x, y : np.column_stack((x,y)), lambda x, y : x * y]:
    for model in [tsne]:
        Z1, Z2 = D['rZ1'], D['rZ2']
        if Z1.ndim != 2:
            Z1 = np.reshape(Z1,(N,-1))
            Z2 = np.reshape(Z2,(N,-1))
        RAND_Z = np.random.normal(size=Z1.shape)
        
        D1 = tsne.fit_transform(Z1)
        D2 = tsne.fit_transform(Z1)
        RAND_D = tsne.fit_transform(RAND_Z)
        
        ax1.scatter(D1[:,0], D1[:,1], c=K)
        ax2.scatter(RAND_D[:,0], RAND_D[:,1], c=K)
        plt.show()
        