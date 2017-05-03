import numpy as np
import matplotlib.pyplot as plt
import collections
import itertools

LABELS = np.array(["Trap.","R. Tri.","Hex.","E. Tri.","Square"])
#SHAPE_ORDER = np.array([2, 3, 0, 4, 1]) # shape order from original experiment
SHAPE_ORDER = np.array([2, 4, 3, 0, 1]) # arranged by sym. order

def argsort_matrix(X_,x_argsort=None):
    x_ = np.diag(X_)
    if x_argsort is None:
        x_argsort = np.argsort(x_)[::-1]
    X = np.zeros_like(X_)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            i_ = x_argsort[i]
            j_ = x_argsort[j]
            X[i,j] = X_[i_,j_]

    x = x_[x_argsort]
    return X, x_argsort

def plot_matrix_helper(X,labels,ax,normalize=False,plot_zeros=True,k=0,thresh=0.6,cmap=plt.cm.Reds):
    labels = [None] + list(labels)
    ax.imshow(X, interpolation='nearest', cmap=cmap)
            #if name is not None:
                #ax.set_title(name)    

    if normalize:
        X = X.astype('float') / X.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = thresh * X.max()
    for i, j in itertools.product(range(X.shape[0]), range(X.shape[1])):
        if not plot_zeros and X[i,j] == 0:
                continue
        ax.text(j, i, '{:.2f}'.format(X[i, j]),
                horizontalalignment="center",
                color="white" if X[i, j] > thresh else "black")

    ax.set_xticklabels(labels)
    if k == 0:
        ax.set_ylabel('Ground Truth',fontweight='bold')
        ax.set_xlabel('Prediction',fontweight='bold')

        ax.set_yticklabels(labels)

def plot_matrix(cms, classes, labels,
                          names=["No Cursor", "Grabbing"],
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    #plt.colorbar()
    labels = [None] + list(labels)
    tick_marks = np.arange(len(classes))

    f, axs = plt.subplots(1,len(cms), figsize=(10,10))
    if not isinstance(axs,collections.Iterable):
        axs = [axs]
    for k, (cm, ax, name) in enumerate(zip(cms, axs, names)):
        plot_matrix_helper(cm, labels, ax, normalize= normalize,
                           cmap = cmap)
        #ax.imshow(cm, interpolation='nearest', cmap=cmap)
        #if name is not None:
            #ax.set_title(name)
        #ax.set_xticks(tick_marks)
        #ax.set_yticks(tick_marks)

        #if normalize:
            #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            #print("Normalized confusion matrix")
        #else:
            #print('Confusion matrix, without normalization')

        #thresh = cm.max() / 2.
        #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            #ax.text(j, i, '{:.2f}'.format(cm[i, j]),
                    #horizontalalignment="center",
                    #color="white" if cm[i, j] > thresh else "black")

        #ax.set_xticklabels(labels)
        #if k == 0:
            #ax.set_ylabel('Ground Truth',fontweight='bold')
            #ax.set_xlabel('Prediction',fontweight='bold')

            #ax.set_yticklabels(labels)

