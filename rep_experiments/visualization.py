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
parser.add_argument('--encoding_time',type=str,default='0')
#parser.add_argument('--encoding_times',type=str,default=['16-11-11-08-31PM']) # position is NOT varied.

parser.add_argument('--use_pca',type=bool,default=0)
parser.add_argument('--use_tsne',type=bool,default=0)
parser.add_argument('--N',type=int,default=None)
args = parser.parse_args()

Zet = []
Ks = []
Ys = []
Ps = []
Ss = []
As = []

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
    T = np.array([[1.,0,0], # tapezoid
                  [0,1.,0], # RightTri
                  [0,0,1.], # Hexagon
                  [1.,1.,0], # Tri
                  [0,1.,1.], # Rect
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

color_p, color_k = {}, {}
#for i, et in enumerate(args.encoding_times):
for i, et in enumerate(list(reversed(['l1_flat_encodings','l2_flat_encodings','l3_flat_encodings',
                        'value_hid_encodings','adv_hid_encodings']))):
    encoding_saver = Saver(time=args.encoding_time,path='{}/{}'.format(DATADIR,'enco_simi_data'))
    simi_data = encoding_saver.load_dictionary(0,'simi_data')
    #encodings = encoding_saver.load_dictionary(0, 'adv_hid_encodings')
    encodings = encoding_saver.load_dictionary(0, et)    
    
    #K = simi_data['SHAPES1'][p]
    #Y = simi_data['Y'][p]
    #S = simi_data['SIZES1'][p]
    #A = simi_data['ANGLES1'][p]
    #Ks.append(K)
    
    
    #encodings = encoding_saver.load_dictionary(0, 'l3_flat_encodings')
    #encodings = encoding_saver.load_dictionary(0, 'adv_hid_encodings')
    #encodings = encoding_saver.load_dictionary(0, 'value_hid_encodings')
    if i == 0:
        if args.N is not None:
            try:
                p = np.random.choice(encodings['rZ1'].shape[0],args.N,replace=False)
            except ValueError:
                p = np.arange(0,encodings['rZ1'].shape[0]) 
        else:
            p = np.arange(0,encodings['rZ1'].shape[0])
        
    N = len(p)
        
    P = simi_data['POSITS1'][p]
    K = simi_data['SHAPES1'][p]
    
    C1 = color_from_2D(P)
    #C2 = color_from_1D(S)
    #C3 = color_from_1D(A)
    C4 = color_from_int(K)    
        
    color_p[et] = C1
    color_k[et] = C4
    Zet.append((et,encodings['rZ1'][p]))

#C = np.array([[1,0,0],
              #[0,0,1]])

pca = PCA(n_components=100)
tsne = TSNE(n_components=2)

markers = ["D","H","s","o","^","2","3","4","8"]

title_color = [("pos.",color_k),("shape",color_k)]

#title_color = filter(lambda x : not np.isnan(x[1].sum()),title_color)
f, axs = plt.subplots(len(Zet),len(title_color),figsize=(5,15))
#f.set_size_inches(5, 50)

if axs.ndim == 1:
    axs= axs[...,None]

DD = []
for model in [None, pca]:
    D = [] 
    for i, (name,z) in enumerate(Zet):
        # create the model
        #if args.use_pca:
            #model = pca
        #if args.use_tsne:
            #model = tsne
        #else:
            #model = None
    
        # fit the model
        z = np.reshape(z,(N,-1))
        if type(model) == PCA:
            model.fit(z)
            M = model.components_.T[:,[0,1]]
            zz= np.dot(z,M)
        elif type(model) == TSNE:
            zz = model.fit_transform(z)
        elif model is None:
            Q, _ = np.linalg.qr(np.random.randn(*((z.T).shape)))
            zz = np.dot(z,Q[:,[0,1]])
        D.append((name,zz))
    DD.append(D)
        
titles = {
    "l1_flat_encodings": "C1",
    "l2_flat_encodings": "C2",
    "l3_flat_encodings": "C3",
    "adv_hid_encodings": "FA",
    "value_hid_encodings": "FV",
}   
for j, (axrow, (title,color)) in enumerate(zip(axs.T, title_color)):
    for i, ax in enumerate(axrow):
        name = DD[0][i][0]
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])        
        #ax.set_xlabel(None)
        if i == 0:
            if j == 0:
                ylab = "Random Projection"
            if j == 1:
                ylab = "PCA"
            D = DD[j]
            ax.set_title(ylab)
        
        if j == 0:
            ax.set_ylabel(titles[name])
            
        if False:
            for k in np.unique(K):
                ixs = K == k
                
                ax.set_title(title)
                #ax.scatter(D[i][ixs,0], D[i][ixs,1], s=30, c=color[ixs])
                ax.scatter(D[i][1][ixs,0], D[i][1][ixs,1], c=color[ixs])
                
                ax.hold(True)
        else:
            #ax.set_title(title)
            ax.scatter(D[i][1][:,0],D[i][1][:,1],c=color[name])
            ax.hold(True)
        

plt.show()

halt= True
    #ax2.scatter(RAND_D[:,0], RAND_D[:,1], m=['o']*len(K))
            