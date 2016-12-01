from shapesorting import *
from sandbox.util import Saver

import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt

import argparse
import numpy as np
import sklearn as sk
import sklearn.metrics as mets

from sklearn.decomposition import *
from sklearn.lda import LDA
from sklearn.manifold import TSNE

np.random.seed(456)

parser = argparse.ArgumentParser()
#parser.add_argument('--encoding_time',type=str,default='16-11-06-nopos')
#parser.add_argument('--encoding_time',type=str,default='16-11-06_w_pos')
parser.add_argument('--encoding_times',type=str,default=['16-11-11-07-18PM'])
#parser.add_argument('--encoding_times',type=str,default=['16-11-11-08-31PM']) # position is NOT varied.

parser.add_argument('--use_pca',type=bool,default=False)
parser.add_argument('--use_tsne',type=bool,default=False)
parser.add_argument('--N',type=int,default=None)
args = parser.parse_args()

Z = []
for et in args.encoding_times:
    encoding_saver = Saver(time=et,path='{}/{}'.format(DATADIR,'enco_simi_data'))
    simi_data = encoding_saver.load_dictionary(0,'simi_data')
    #encodings = encoding_saver.load_dictionary(0, 'adv_hid_encodings')
    encodings = encoding_saver.load_dictionary(0, 'l3_flat_encodings')
    encodings = encoding_saver.load_dictionary(0, 'adv_hid_encodings')
    encodings = encoding_saver.load_dictionary(0, 'value_hid_encodings')
    if args.N is not None:
        try:
            p = np.random.choice(encodings['rZ1'].shape[0],args.N,replace=False)
        except ValueError:
            p = np.arange(0,encodings['rZ1'].shape[0]) 
    else:
        p = np.arange(0,encodings['rZ1'].shape[0])
    Z.append(encodings['rZ1'][p])

C = np.array([[1,0,0],
              [0,0,1]])
K = simi_data['SHAPES1'][p]
Y = simi_data['Y'][p]
P = simi_data['POSITS1'][p]
S = simi_data['SIZES1'][p]
A = simi_data['ANGLES1'][p]

N = len(Y)

pca = PCA(n_components=2)
tsne = TSNE(n_components=2)

if args.use_pca:
    model = pca
if args.use_tsne:
    model = tsne

def color_from_2D(X):
    X = X.astype('float32')
    X -= X.min()
    X /= X.max()
    try:
        return np.column_stack((X,np.zeros_like(X[:,0])[...,None]))
    except:
        return X

def color_from_1D(X):
    X = X.astype('float32')
    X -= X.min()
    X /= X.max()
    #return mpcolors.hsv_to_rgb(np.column_stack([0.5 * X,np.ones_like(X),X]))
    return np.column_stack([X,1-X,abs(0.5-X)])

def color_from_int(X):
    T = np.array([[1.,0,0],
                  [0,1.,0],
                  [0,0,1.],
                  [1.,1.,0],
                  [0,1.,1.],
                  [1.,0,1.],
                  [0.5,0.2,0.9]])
    return T[X]

def cyclic_color_from_1D(X):
    X = X.astype('float32')
    X -= 180
    X = np.abs(X)
    
    X -= X.min()
    X /= X.max()
    return np.column_stack([X,1-X,np.zeros_like(X)])


C1 = color_from_2D(P)
C2 = color_from_1D(S)
C3 = color_from_1D(A)
C4 = color_from_int(K)

markers = ["D","H","s","o","^","2","3","4","8"]
#for merge in [lambda x, y : np.column_stack((x,y)), lambda x, y : x * y]:
    #for model in [tsne]:
    
#title_color = [("pos.",C1),("size",C2),("ang.",C3),("shape",C4)]
title_color = [("pos.",C1),("shape",C4)]

title_color = filter(lambda x : not np.isnan(x[1].sum()),title_color)
f, axs = plt.subplots(len(title_color),1)
if axs.ndim == 1:
    axs= axs[...,None]
    
# reduce dims

#Z1, _ = D[0]['rZ1']
#Z2, _ = D[1]['rZ1']

#Z = [D[i]['rZ1'] for i in range(len(args.encoding_times))]
D = []
for i in range(len(Z)):
    Z[i] = np.reshape(Z[i],(N,-1))
    D.append(model.fit_transform(Z[i]))

#if Z1.ndim != 2:
    #Z1 = np.reshape(Z1,(N,-1))
#if Z2.ndim != 2:
    #Z2 = np.reshape(Z2,(N,-1))    

#D1 = tsne.fit_transform(Z1)
#D2 = tsne.fit_transform(Z2)
    
for axrow, (title,color) in zip(axs, title_color):
    for i, ax in enumerate(axrow):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])        
        #ax.set_xlabel(None)
        #ax.set_ylabel(None)
        
        if False:
            for k in np.unique(K):
                ixs = K == k
                
                ax.set_title(title)
                #ax.scatter(D[i][ixs,0], D[i][ixs,1], s=30, c=color[ixs])
                ax.scatter(D[i][ixs,0], D[i][ixs,1], c=color[ixs])
                
                ax.hold(True)
        else:
            #ax.set_title(title)
            ax.scatter(D[i][:,0],D[i][:,1],c=color)
            ax.hold(True)

plt.show()
    #ax2.scatter(RAND_D[:,0], RAND_D[:,1], m=['o']*len(K))
            