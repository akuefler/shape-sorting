import argparse
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from config import DATADIR

from util import Saver

parser = argparse.ArgumentParser()

## includes holes
#parser.add_argument("--data_time",type=str,default="17-01-19-07-59PM") # no grab
#parser.add_argument("--data_time",type=str,default="17-01-19-09-27PM") # grab 1
#parser.add_argument("--data_time",type=str,default="17-01-19-09-29PM") # grab 2

## no holes
#parser.add_argument("--data_time",type=str,default="17-01-19-09-35PM") # no grab
#parser.add_argument("--data_time",type=str,default="17-01-19-09-36PM") # grab 1
#parser.add_argument("--data_time",type=str,default="17-01-19-09-37PM") # grab 2

## Fixed Position
#parser.add_argument("--data_time",type=str,default="17-01-19-22-34-07-174659") # no grab
parser.add_argument("--data_time",type=str,default="17-01-19-22-34-56-056347") # grab 1
#parser.add_argument("--data_time",type=str,default="17-01-19-22-35-45-287050") # grab 2

## Fixed Position, No Holes
#parser.add_argument("--data_time",type=str,default="17-01-19-22-36-33-368248") # no grab
#parser.add_argument("--data_time",type=str,default="17-01-19-22-37-16-244572") # grab 1
#parser.add_argument("--data_time",type=str,default="17-01-19-22-37-58-907670") # grab 2


parser.add_argument("--encodings",nargs="+",type=str,default=["Z_l1_flat","Z_l2_flat","Z_l3_flat","Z_value_hid","Z_adv_hid"])

parser.add_argument("--N",type=int,default=1000)
parser.add_argument("--model",type=str,default="pca")
parser.add_argument("--color_fn",type=str,default="shape")
args = parser.parse_args()

assert args.color_fn in ["shape","center","delta"]

data_saver = Saver(time=args.data_time,path='{}/{}'.format(DATADIR,'scene_and_enco_data'))

data = data_saver.load_dictionary(0,"data")
blocks = data_saver.load_recursive_dictionary("blocks")
holes = data_saver.load_recursive_dictionary("holes")

def color_from_1D(X):
    X = X.astype('float32')
    X -= X.min()
    X /= X.max()
    #return mpcolors.hsv_to_rgb(np.column_stack([0.5 * X,np.ones_like(X),X]))
    return np.column_stack([X,1-X,abs(0.5-X)])

def color_from_2D(X):
    X = X.astype('float32')
    X -= X.min()
    X /= X.max()
    try:
        return np.column_stack((X,np.zeros_like(X[:,0])[...,None]))
    except:
        return X

def color_from_int(X):
    T = np.array([[1.,0,0], # tapezoid
                  [0,1.,0], # RightTri
                  [0,0,1.], # Hexagon
                  [1.,1.,0], # Tri
                  [0,1.,1.], # Rect
                  [1.,0,1.],
                  [0.5,0.2,0.9]])
    return T[X]

NN = data["X"].shape[0]

BLOCK_CENTER = np.row_stack([np.array(blocks['{:0{}}'.format(i,len(str(NN)))]['center']) for i in range(NN)])
BLOCK_SHAPE = np.concatenate([np.array(blocks['{:0{}}'.format(i,len(str(NN)))]['shape']) for i in range(NN)])
b_ix = [list(holes['{:0{}}'.format(i,len(str(NN)))]['shape']).index(blocks['{:0{}}'.format(i,len(str(NN)))]['shape'])
        for i in range(NN)]
BLOCK_DELTAS = np.array([np.linalg.norm(holes['{:0{}}'.format(i,len(str(NN)))]['center'][b_ix[i]] - \
                                              blocks['{:0{}}'.format(i,len(str(NN)))]['center']) for i in range(NN)])

color_fns = {"center":color_from_2D(BLOCK_CENTER),
             "shape":color_from_int(BLOCK_SHAPE),
             "delta":color_from_1D(BLOCK_DELTAS)}

C = color_fns[args.color_fn]

models = {"pca":PCA(),"tsne":TSNE()}
model = models[args.model]

p = np.random.choice(range(NN),args.N,replace=False)
f, axs = plt.subplots(1,len(args.encodings))
for layer, ax in zip(args.encodings,axs):
    G = model.fit_transform(data[layer][p])
    ax.scatter(G[:,0],G[:,1],c=C[p],s=100)
    
plt.show()

halt= True

