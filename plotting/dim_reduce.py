import argparse
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
import matplotlib.pyplot as plt

from config import DATADIR

from util import Saver
from config import *

parser = argparse.ArgumentParser()

matplotlib.rcParams.update({'font.size': 22})

## includes holes
#parser.add_argument("--data_time",type=str,default="17-01-19-07-59PM") # no grab
#parser.add_argument("--data_time",type=str,default="17-01-19-09-27PM") # grab 1
#parser.add_argument("--data_time",type=str,default="17-01-19-09-29PM") # grab 2

#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-19-07-59PM",
                                                                  #"17-01-19-09-27PM",
                                                                  #"17-01-19-09-29PM"])

## no holes
#parser.add_argument("--data_time",type=str,default="17-01-19-09-35PM") # no grab
#parser.add_argument("--data_time",type=str,default="17-01-19-09-36PM") # grab 1
#parser.add_argument("--data_time",type=str,default="17-01-19-09-37PM") # grab 2

#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-19-09-35PM",
                                                                  #"17-01-19-09-36PM",
                                                                  #"17-01-19-09-37PM"])

## Fixed Position
parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-19-22-34-07-174659",
                                                                  "17-01-19-22-34-56-056347",
                                                                  ])
                                                                  #"17-01-19-22-35-45-287050"])

## Fixed Position, No Holes
#parser.add_argument("--data_time",type=str,default="17-01-19-22-36-33-368248") # no grab
#parser.add_argument("--data_time",type=str,default="17-01-19-22-37-16-244572") # grab 1
#parser.add_argument("--data_time",type=str,default="17-01-19-22-37-58-907670") # grab 2

#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-19-22-36-33-368248",
                                                                  #"17-01-19-22-37-16-244572",
                                                                  #"17-01-19-22-37-58-907670"])
                           
## Holes, Enumerated
#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-20-21-12-51-477494",
                                                                  #"17-01-20-21-13-10-048480",
                                                                  #"17-01-20-21-13-35-095721"])
## No Holes, Enumerated
#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-20-21-55-33-448895",
                                                                  #"17-01-20-21-56-02-187089",
                                                                  #"17-01-20-21-56-21-432451"])

parser.add_argument("--encodings",nargs="+",type=str,default=["Z_l1_flat","Z_l2_flat","Z_l3_flat","Z_value_hid","Z_adv_hid"])

parser.add_argument("--N",type=int,default=1000)
parser.add_argument("--model",type=str,default="pca")
parser.add_argument("--color_fn",type=str,default="shape")
args = parser.parse_args()

assert args.color_fn in ["shape","center","delta"]

data_savers = [Saver(time=data_time,path='{}/{}'.format(DATADIR,'scene_and_enco_data'))
               for data_time in args.data_times]

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

f, axs = plt.subplots(len(data_savers),len(args.encodings), figsize=(2.5 * 10,10))
xlabels = ["Conv1", "Conv2", "Conv3", "Val.", "Adv."]
ylabels = ["No Cursor", "Grabbing"]

for j, data_saver in enumerate(data_savers):

    data = data_saver.load_dictionary(0,"data")
    blocks = data_saver.load_recursive_dictionary("blocks")
    holes = data_saver.load_recursive_dictionary("holes")
    
    NN = data["X"].shape[0]
    
    BLOCK_CENTER = np.row_stack([np.array(blocks['{:0{}}'.format(i,len(str(NN)))]['center']) for i in range(NN)])
    BLOCK_SHAPE = np.concatenate([np.array(blocks['{:0{}}'.format(i,len(str(NN)))]['shape']) for i in range(NN)])
    
    #b_ix = [list(holes['{:0{}}'.format(i,len(str(NN)))]['shape']).index(blocks['{:0{}}'.format(i,len(str(NN)))]['shape'])
            #for i in range(NN)]
    #BLOCK_DELTAS = np.array([np.linalg.norm(holes['{:0{}}'.format(i,len(str(NN)))]['center'][b_ix[i]] - \
                                                  #blocks['{:0{}}'.format(i,len(str(NN)))]['center']) for i in range(NN)])
    
    color_fns = {"center":color_from_2D(BLOCK_CENTER),
                 "shape":color_from_int(BLOCK_SHAPE)}
                 #"delta":color_from_1D(BLOCK_DELTAS)}
    
    C = color_fns[args.color_fn]
    
    models = {"pca":PCA(),"tsne":TSNE()}
    model = models[args.model]
    
    p = np.random.choice(range(NN),args.N,replace=False)
    print(len(p))
    for i, (layer, ax) in enumerate(zip(args.encodings,axs[j])):
        if j == len(data_savers) - 1:
            ax.set_xlabel(xlabels[i],fontweight="bold")
        if i == 0:
            ax.set_ylabel(ylabels[j],fontweight="bold")
        
        G = model.fit_transform(data[layer][p])
        handles = ax.scatter(G[:,0],G[:,1],c=C[p],s=85)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#plt.show()
plt.tight_layout()        
plt.savefig("{}pca.pdf".format(FIGDIR),bbox_inches='tight',format='pdf',dpi=300)

