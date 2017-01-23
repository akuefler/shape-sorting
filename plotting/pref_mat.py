from matplotlib import gridspec

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import h5py

matplotlib.rcParams.update({'font.size': 22})

with h5py.File("preference_mat.h5","r") as hf:
    X_ = hf['C'][...]

ylabel = "Winner"
xlabel = "Loser"
if True:
    X_ = X_.T
    temp = xlabel
    xlabel = ylabel
    ylabel = temp
    
x_ = X_.sum(axis=0)
x_argsort = x_.argsort()[::-1]

if True:
    X_ = (X_.T / X_.sum(axis=1)).T

if True:
    X = np.zeros_like(X_)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            i_ = x_argsort[i]
            j_ = x_argsort[j]
            X[i,j] = X_[i_,j_]
            
    x = x_[x_argsort]
else:
    x_argsort = np.array(range(5))
    x = x_
    X = X_

##
from matplotlib.ticker import NullFormatter
nullfmt = NullFormatter()         # no labels

# definitions for the axes
labels = np.array(["Trap.","R. Tri.","Hex.","E. Tri.","Square"])
hw = 0.65

left, width = 0.2, hw
bottom, height = 0.1, hw
bottom_h = left_h = bottom + hw

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
fsize = 10
plt.figure(1, figsize=(fsize, fsize))

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axScatter.matshow(X,cmap="Greys")

axScatter.xaxis.tick_bottom()
axScatter.set_xticklabels([""] + list(labels[x_argsort]))
axScatter.set_yticklabels([""] + list(labels[x_argsort]))

axScatter.set_xlabel(xlabel, fontsize=28)
axScatter.set_ylabel(ylabel, fontsize=28)

binwidth = 1

lim0, lim1 = axScatter.get_xlim()

bins = np.arange(lim0, lim1 + binwidth, binwidth)

axHistx.bar(range(len(x)),x, width=1.0, facecolor="white", edgecolor='black', linewidth=5)
axHistx.set_ylim((axHistx.get_ylim()[0],axHistx.get_ylim()[1] + 5))

plt.show()

halt= True