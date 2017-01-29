from util import Saver
from config import *

from plotting import *

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import h5py
import argparse

matplotlib.rcParams.update({'font.size': 22})

parser = argparse.ArgumentParser()
parser.add_argument("--data_time",type=str,default="17-01-28-19-07-13-805742")

args = parser.parse_args()

data_saver = Saver(time=args.data_time, path='{}/{}'.format(DATADIR,'pref_results'))
data = data_saver.load_dictionary(0,"data")

ylabel = "Loser"
xlabel = "Winner"
X = np.nan_to_num(data['victors'] / data['totals'])
x = data['victors'].sum(axis=0) / data['totals'].sum(axis=0)

x_argsort = range(5)
if True:
    X, x_argsort = argsort_matrix(X, x_argsort=np.argsort(x)[::-1])
    x = x[x_argsort]

##
from matplotlib.ticker import NullFormatter
nullfmt = NullFormatter()         # no labels

# definitions for the axes
hw = 0.4

left, width = 0.3, hw
bottom, height = 0.1, hw
bottom_h = left_h = bottom + hw

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, hw]
fsize = 15
f = plt.figure(1, figsize=(fsize, fsize))

axMat = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
#axScatter.matshow(X,cmap="Greys")
plot_matrix_helper(X, xlabel, axMat, thresh = 0.5, plot_zeros = False)

axMat.xaxis.tick_bottom()
axMat.set_xticklabels([""] + list(LABELS[x_argsort]))
axMat.set_yticklabels([""] + list(LABELS[x_argsort]))

axMat.set_xlabel(xlabel, fontsize=28)
axMat.set_ylabel(ylabel, fontsize=28)

binwidth = 1

lim0, lim1 = axMat.get_xlim()

bins = np.arange(lim0, lim1 + binwidth, binwidth)

axHistx.bar(range(len(x)),x, width=1.0, facecolor="white",edgecolor="black", linewidth=5)
#axHistx.set_ylim((axHistx.get_ylim()[0],axHistx.get_ylim()[1] + 5))
axHistx.set_ylim(0,1)

axHistx.grid(b=True, which='major')
axHistx.grid(b=True, which='minor')

#plt.show()
plt.savefig("{}pref_mat.png".format(FIGDIR),bbox_inches='tight')

halt= True