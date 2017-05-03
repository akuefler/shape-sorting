import argparse
import numpy as np

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import DATADIR

from util import Saver
from plotting import *
from config import *

from sklearn.metrics import accuracy_score, confusion_matrix

import itertools
import h5py

parser = argparse.ArgumentParser()

matplotlib.rcParams.update({'font.size': 22})


"""
Assorted datasets with different hyperparameters.
"""
## includes holes
#parser.add_argument("--data_time",type=str,default="17-01-19-07-59PM") # no grab
#parser.add_argument("--data_time",type=str,default="17-01-19-09-27PM") # grab 1
#parser.add_argument("--data_time",type=str,default="17-01-19-09-29PM") # grab 2

#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-19-07-59PM",
#                                                                  "17-01-19-09-27PM"])
                                                                  #"17-01-19-09-29PM"])

## no holes
#parser.add_argument("--data_time",type=str,default="17-01-19-09-35PM") # no grab
#parser.add_argument("--data_time",type=str,default="17-01-19-09-36PM") # grab 1
#parser.add_argument("--data_time",type=str,default="17-01-19-09-37PM") # grab 2

#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-19-09-35PM",
                                                                  #"17-01-19-09-36PM",
                                                                  #"17-01-19-09-37PM"])


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

"""
used these for the discriminator experiment.
"""
## TRAINED network
parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-20-23-17-53-444875",
                                     "17-01-20-23-26-23-488455"])
                                     ##"17-01-20-23-33-47-043926"])
## UNTRAINED network
#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-29-14-21-59-644400",
#                                                                  "17-01-29-14-30-54-397244"])

"""
used these data for the paper.
"""
## Fixed Position
#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-19-22-34-07-174659",
#                                                                  "17-01-19-22-34-56-056347",
#                                                                  ])

                                                                  #"17-01-19-22-35-45-287050"])


parser.add_argument("--encodings",nargs="+",type=str,default=["Z_l1_flat","Z_l2_flat","Z_l3_flat","Z_value_hid","Z_adv_hid"])

parser.add_argument("--N",type=int,default=1000)
parser.add_argument("--color_fn",type=str,default="shape")
parser.add_argument("--save",type=int,default=1)
parser.add_argument("--seed",type=int,default=456)
parser.add_argument("--normalize",type=int,default=0)
parser.add_argument("--reduced_dim",type=int,default=300)
parser.add_argument("--viz_w_lda",type=int,default=0)

args = parser.parse_args()

assert args.color_fn in ["shape","center","delta"]


data_savers = [Saver(time=data_time,path='{}/{}'.format(DATADIR,'scene_and_enco_data'))
               for data_time in args.data_times]

PERM = np.row_stack(list(itertools.permutations(range(5))))
def cluster_to_label(centroids,labels):
    accs = []
    acc_vectors = []
    EYE = np.eye(5)
    for perm in PERM:
        c = perm[centroids]
        #acc_vector = np.mean(EYE[labels] - EYE[c] > 0, axis=0)
        acc_ = accuracy_score(labels,c)
        cm = confusion_matrix(labels,c).astype('float32')
        #cm /= cm.sum(axis=1)[...,np.newaxis]
        acc_vector = np.diag(cm) / cm.sum(axis=1)

        acc_vectors.append(acc_vector)
        acc = np.diag(cm).sum() / cm.sum() #np.mean(acc_vector)
        assert np.allclose(acc, acc_)
        accs.append(np.mean(acc_vector))
        #accs.append(accuracy_score(c, labels))
    i = np.argmax(accs)
    return PERM[i][centroids], acc_vectors[i]

def compute_coherence(centroids, datapoints, labels):
    D = []
    for label in np.unique(labels):
        d = np.linalg.norm(datapoints[labels == label] - centroids[label],axis=1)
        D.append(np.mean(d))
    D = np.array(D)

    return D

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
    T = np.array([[1.,1.,1.],
                  [1.,0,0], # tapezoid
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

cohere_0 = []
acc_0 = []
for j, data_saver in enumerate(data_savers):
    print("######")
    print(data_saver.load_args())

    data = data_saver.load_dictionary(0,"data")
    blocks = data_saver.load_recursive_dictionary("blocks")
    holes = data_saver.load_recursive_dictionary("holes")
    NN = data["X"].shape[0]
    BLOCK_CENTER = np.row_stack([np.array(blocks['{:0{}}'.format(i,len(str(NN)))]['center']) for i in range(NN)])
    BLOCK_SHAPE = np.concatenate([np.array(blocks['{:0{}}'.format(i,len(str(NN)))]['shape']) for i in range(NN)])
    color_fns = {"center":color_from_2D(BLOCK_CENTER),
                 "shape":color_from_int(BLOCK_SHAPE)}
    C = color_fns[args.color_fn]
    pca = PCA()
    lda = LDA()

    np.random.seed(args.seed)
    p = np.random.choice(range(NN),args.N,replace=False)

    cohere_1 = []
    acc_1 = []
    for i, (layer, ax) in enumerate(zip(args.encodings,axs[j])):
        # label axes
        if j == len(data_savers) - 1:
            ax.set_xlabel(xlabels[i],fontweight="bold")
        if i == 0:
            ax.set_ylabel(ylabels[j],fontweight="bold")

        # sample activations and normalize
        X = data[layer][p]
        if args.normalize:
            X -= X.mean(axis=0)
            X /= (X.std(axis=0) + 1e-10)

        # project data
        if args.reduced_dim > 0:
            G = pca.fit_transform(X)[:,:args.reduced_dim]
        else:
            print("Warning: Skipping PCA")
            G = X
        #G = (G - G.mean(axis=0))/(G.std(axis=0) + 1e-8)

        # cluster projected data
        kmeans = KMeans(n_clusters = 5)
        kmeans.fit(G)

        l = BLOCK_SHAPE[p]
        y, acc_vector = cluster_to_label(kmeans.predict(G),l)
        acc = np.mean(acc_vector)

        print("Dataset: {}, Layer: {}, Acc: {}".format(j,i,acc))
        cohere_vector = compute_coherence(kmeans.cluster_centers_,G,l)
        cohere_1.append(cohere_vector)
        acc_1.append(acc_vector)
        print("Coherence: {}; Inertia: {}".format(
            cohere_vector,kmeans.inertia_)
            )

        # use LDA just for the visulization step.
        if args.viz_w_lda:
            G = lda.fit_transform(X, l)
        correct = (l == y).astype('int32')
        G_correct = G[correct == 1]
        c_correct = color_from_int(l[correct == 1] + 1)
        G_wrong = G[correct == 0]
        c_wrong = color_from_int(l[correct == 0] + 1)

        handles = ax.scatter(G_wrong[:,0], G_wrong[:,1], facecolors='none', edgecolor=c_wrong, s=85)
        handles = ax.scatter(G_correct[:,0], G_correct[:,1], facecolors=c_correct, s=85)

        #c = color_from_int(np.array(l == y).astype('int32') * (y+1))
        #cl = color_from_int((l+1))
        #handles = ax.scatter(G[:,0],G[:,1], color=c, alpha=.5, edgecolor=cl, s=85)

        #cy = color_from_int(y + 1)
        #cl = color_from_int(l + 1)
        #handles = ax.scatter(G[:,0],G[:,1], color=cy, edgecolor=cl, s=85)
        #handles = ax.scatter(G[:,0], G[:,1], color=cy, edgecolor=cl, s=85, linewidth=3)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    cohere_0.append(cohere_1)
    acc_0.append(acc_1)

# grab x layer x shape
#SPREAD_MAT = np.array(cohere_0)
#ACC_MAT = np.array(acc_0)
#with h5py.File("k_means_meta.h5","a") as hf:
#    hf.create_dataset("spread", data=SPREAD_MAT)
#    hf.create_dataset("acc", data=ACC_MAT)

# add a legend
patches = []
for i, name in enumerate(LABELS):
    color = color_from_int(i + 1)
    patches.append(mpatches.Patch(color= color_from_int(i + 1), label= name))
patches = np.array(patches)[SHAPE_ORDER]

#plt.legend(loc='upper center', bbox_to_anchor=(-5, -0.05), handles=patches, ncol=len(patches))
plt.sca(axs[-1,2])
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), handles=list(patches), ncol=len(patches))

if args.save:
    plt.savefig("{}/kmeans.pdf".format(FIGDIR),bbox_inches='tight',format='pdf',dpi=300)
else:
    plt.show()

