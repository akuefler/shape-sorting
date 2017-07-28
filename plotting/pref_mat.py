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
#parser.add_argument("--data_time",type=str,default="17-01-30-22-03-52-279284")
#parser.add_argument("--data_time",type=str,default="17-04-22-12-10-45-653249")

#parser.add_argument("--data_time",type=str,default="17-04-22-22-10-22-293106") # test_ep is 0.1
parser.add_argument("--data_time",type=str,default="17-04-22-19-31-09-444847") # test_ep is 0.0
parser.add_argument("--save",type=int,default=0)

args = parser.parse_args()

data_saver = Saver(time=args.data_time, path='{}/{}'.format(DATADIR,'pref_results'))
data = data_saver.load_dictionary(0,"data")
try:
    exp_args = data_saver.load_dictionary(0,"args")
except:
    exp_args = None

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
axHistxHeight = 0.2

left, width = 0.3, hw
bottom, height = 0.1, hw
bottom_h = bottom + hw + 0.01

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, axHistxHeight]
fsize = 15
f = plt.figure(1, figsize=(fsize, fsize))

axMat = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
#axScatter.matshow(X,cmap="Greys")
plot_matrix_helper(X, xlabel, axMat, thresh = 0.5, plot_diag = False)

axMat.xaxis.tick_bottom()
axMat.set_xticklabels([""] + list(LABELS[x_argsort]))
axMat.set_yticklabels([""] + list(LABELS[x_argsort]))

axMat.set_xlabel(xlabel, fontsize=28)
axMat.set_ylabel(ylabel, fontsize=28)

binwidth = 1
lim0, lim1 = axMat.get_xlim()

bins = np.arange(lim0, lim1 + binwidth, binwidth)

axHistx.bar(range(len(x)),x, width=1.0, facecolor="white",edgecolor="black",
        linewidth=4)
for i, x_val in enumerate(x):
    axHistx.text(i + 0.5, 0.05, '{:.2f}'.format(x_val),
            horizontalalignment="center", color="black")

#axHistx.set_ylim((axHistx.get_ylim()[0],axHistx.get_ylim()[1] + 5))
axHistx.set_yticklabels([])
axHistx.set_ylim(0,1)
x_lo, x_hi = axHistx.get_xlim()
pad = 0.025
axHistx.set_xlim(x_lo - pad, x_hi + pad)

axHistx.grid(b=True, which='major')
axHistx.grid(b=True, which='minor')

if args.save:
    plt.savefig("{}/pref_mat.pdf".format(FIGDIR),bbox_inches='tight',format='pdf',dpi=800)
else:
    plt.show()

