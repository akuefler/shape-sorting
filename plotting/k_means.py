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
#parser.add_argument("--data_times", nargs="+", type=str,
#        default=["17-01-19-22-34-07-174659",
#                 "17-01-19-22-34-56-056347"])
#                                                                  #"17-01-19-22-35-45-287050"])


#parser.add_argument("--encodings",nargs="+",type=str,default=["Z_l1_flat","Z_l2_flat","Z_l3_flat","Z_value_hid","Z_adv_hid"])
parser.add_argument("--encodings",nargs="+",type=str,default=["Z_l3_flat"])

parser.add_argument("--N",type=int,default=2500)
parser.add_argument("--color_fn",type=str,default="shape")
parser.add_argument("--save",type=int,default=0)
parser.add_argument("--seed",type=int,default=456)
parser.add_argument("--normalize",type=int,default=0)
parser.add_argument("--reduced_dim",type=int,default=300)
parser.add_argument("--color_correct",type=int,default=0)
parser.add_argument("--viz_w_lda",type=int,default=1)
parser.add_argument("--transpose",type=int,default=1)

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

if args.transpose:
    n_cols = len(data_savers)
    n_rows = len(args.encodings)
else:
    n_rows = len(data_savers)
    n_cols = len(args.encodings)
f, axs = plt.subplots(n_rows,n_cols, figsize=(n_cols * 10, n_rows * 10))
one_data = len(data_savers) == 1
one_enco = len(args.encodings) == 1
if one_data and one_enco:
    axs = np.array([[axs]])
elif one_data or one_enco:
    if args.transpose:
        axs = axs[None,...]
    else:
        axs = axs[...,None]

xlabels = ["Conv1", "Conv2", "Conv3", "Val.", "Adv."]
ylabels = ["No Cursor", "Grabbing"]

cohere_0 = []
acc_0 = []
for j, data_saver in enumerate(data_savers):
    print("######")
    print(data_saver.load_args())

    print("loading data ...")
    data = data_saver.load_dictionary(0,"data")
    print("... finished loading data")

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

    for i, layer in enumerate(args.encodings):
        if args.transpose:
            ax = axs[i,j]
        else:
            ax = axs[j,i]

        # label axes
        if args.transpose:
            #if j == len(data_savers) - 1 and n_cols > 1:
            #    ax.set_xlabel(xlabels[i],fontweight="bold")
            #if j == 0 and n_cols > 1:
            ax.set_xlabel(ylabels[j], fontsize= 34, fontweight="bold")

        else:
            if i == len(data_savers) - 1 and n_cols > 1:
                ax.set_xlabel(xlabels[i],fontweight="bold")
            if j == 0 and n_rows > 1:
                ax.set_ylabel(ylabels[j],fontweight="bold")

        # sample activations and normalize
        X = data[layer][p]
        if args.normalize:
            X -= X.mean(axis=0)
            X /= (X.std(axis=0) + 1e-10)

        # project data

        print("fit transform with PCA ...")
        if args.reduced_dim > 0:
            G = pca.fit_transform(X)[:,:args.reduced_dim]
        else:
            print("Warning: Skipping PCA")
            G = X

        # cluster projected data
        l = BLOCK_SHAPE[p]

        print("computing labels ...")
        if args.color_correct:
            kmeans = KMeans(n_clusters = 5)
            kmeans.fit(G)
            y, acc_vector = cluster_to_label(kmeans.predict(G),l)
            acc = np.mean(acc_vector)

            print("Dataset: {}, Layer: {}, Acc: {}".format(j,i,acc))
            cohere_vector = compute_coherence(kmeans.cluster_centers_,G,l)
            cohere_1.append(cohere_vector)
            acc_1.append(acc_vector)
            print("Coherence: {}; Inertia: {}".format(
                cohere_vector,kmeans.inertia_)
                )
            correct = (l == y).astype('int32')
        else:
            correct = (l == l).astype('int32')

        # use LDA just for the visulization step.
        if args.viz_w_lda:
            G = lda.fit_transform(X, l)

        G_correct = G[correct == 1]
        c_correct = color_from_int(l[correct == 1] + 1)
        G_wrong = G[correct == 0]
        c_wrong = color_from_int(l[correct == 0] + 1)

        print("plotting ... ")
        if args.color_correct:
            handles = ax.scatter(G_wrong[:,0], G_wrong[:,1], facecolors='none', edgecolor=c_wrong, s=85)
            handles = ax.scatter(G_correct[:,0], G_correct[:,1], facecolors=c_correct, s=85)
        else:
            handles = ax.scatter(G_correct[:,0], G_correct[:,1],
                    facecolors=c_correct, s=85)

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

    if args.color_correct:
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
#axs[0,0].set_title("Conv 3 Projected Encodings")
if True:
    plt.sca(axs[-1,len(args.encodings)/2])
    #plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), handles=list(patches), ncol=len(patches))
    plt.legend(loc='lower center', bbox_to_anchor=(1.025, -0.2),
            handles=list(patches), ncol=len(patches))
else:
    plt.sca(axs[-1,0])
    plt.legend(loc='center left', bbox_to_anchor=(1, 1.05),
            handles=list(patches))

plt.subplots_adjust(wspace=0.05, hspace=0.05)
if args.save:
    plt.savefig("{}/kmeans.pdf".format(FIGDIR),bbox_inches='tight',format='pdf',dpi=800)
else:
    plt.show()

