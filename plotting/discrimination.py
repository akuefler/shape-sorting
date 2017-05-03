import argparse
import numpy as np

from plotting import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

from config import *

from util import Saver

import itertools
import collections

parser = argparse.ArgumentParser()

#parser.add_argument("--data_time",type=str,default="17-01-28-13-58-54-152302")
parser.add_argument("--data_time",type=str,default="17-04-30-12-29-08-003193")

#parser.add_argument("--baseline_time",type=str,default="17-01-29-14-52-42-083367")
parser.add_argument("--baseline_time",type=str,default="17-04-30-14-07-21-090706")


parser.add_argument("--linewidth",type=int,default=5)
parser.add_argument("--legend_size",type=int,default=25)
parser.add_argument("--val",type=int,default=1)
parser.add_argument("--save",type=int,default=0)

args = parser.parse_args()

data_saver = Saver(time=args.data_time, path='{}/{}'.format(DATADIR,'disc_results'))
baseline_saver = Saver(time=args.baseline_time, path='{}/{}'.format(DATADIR,'disc_results'))

data = data_saver.load_dictionary(0,"data")
exp_args = data_saver.load_args()

baseline = baseline_saver.load_dictionary(0,"data")
MT, MV = data["MT"], data["MV"] # grab conditions x layers x models
BT, BV = baseline["MT"], baseline["MV"]

MT_mean = MV.mean(axis=-1)
MV_mean = MV.mean(axis=-1)

BT_mean = BT.mean(axis=-1)
BV_mean = BV.mean(axis=-1)
best_layer = MV_mean.mean(axis=0).argmax()

matplotlib.rcParams.update({'font.size': 22})

## plot both
#f, axs = plt.subplots(1,2)
#for i, color in enumerate(['r','g','b']):
    #axs[0].plot(MT_mean[i,:],color=color,linewidth=args.linewidth)
    #axs[1].plot(MV_mean[i,:],color=color,linewidth=args.linewidth)
    #axs[0].set_ylim((0.2,0.8))
    #axs[1].set_ylim((0.2,0.8))
if args.val:
    M = MV_mean
    B = BV_mean
    C = data['CV'][0,best_layer].mean(axis=0)
    C_grab = data['CV'][1,best_layer].mean(axis=0)
    C, c_argsort = argsort_matrix(C, x_argsort=SHAPE_ORDER)
    C_grab, c_argsort_ = argsort_matrix(C_grab, x_argsort=SHAPE_ORDER)
    assert np.all(c_argsort == c_argsort_)
else:
    M = MT_mean
    B = BT_mean
    C = data['CT'][0,best_layer].mean(axis=0)
    C_grab = data['CT'][1,best_layer].mean(axis=0)
    C, c_argsort = argsort_matrix(C, x_argsort=SHAPE_ORDER)
    C_grab, c_argsort_ = argsort_matrix(C_grab, x_argsort=SHAPE_ORDER)

#f, (ax_line, ax_mat) = plt.subplots(2,1,figsize=(10,10),sharex=True)
#ax_line = plt.subplot2grid((2,1), (0,0), colspan=1)
#ax_mat = plt.subplot2grid((2,1), (1, 0), colspan=3)

#f = plt.figure(figsize=(10, 20)) 
#gs = gridspec.GridSpec(2, 1, width_ratios=[3,3], height_ratios=[1, 3]) 
#gs.update(wspace=0.025, hspace=0.00005)

#ax_line = plt.subplot(gs[0])
#ax_mat = plt.subplot(gs[1])
#plt.subplots_adjust(hspace=0)

hw = 0.4

left, width = 0.3, hw
bottom, height = 0.1, hw
bottom_h = left_h = bottom + 0.1 + hw

rect_mat = [left, bottom, width, height]
rect_line = [left, bottom_h, width, hw]
fsize = 15
f = plt.figure(1, figsize=(fsize, fsize))

ax_mat = plt.axes(rect_mat)
ax_line = plt.axes(rect_line)

#plt.tight_layout()

plot_matrix_helper(C_grab, LABELS[c_argsort], ax_mat, normalize=True, plot_zeros=True, k=0,
                   thresh=0.6, cmap=plt.cm.Blues)

artists = []
for i, (color, b_color) in enumerate([('r',[1.0,0.5,0.5]),('b',[0.5,0.5,1.0])]):
    artist = ax_line.plot(M[i,:],color=color,linewidth=args.linewidth)
    ax_line.plot(B[i,:],color=b_color,linewidth=args.linewidth,linestyle='--')
    artists.append(artist[0])

xlo, xhi = ax_line.get_xlim()
ax_line.set_xticks(np.arange(xlo, xhi + 1.0))
ax_line.set_xticklabels(['Conv1','Conv2','Conv3','Val.','Adv.'])

ax_line.tick_params(axis="x",pad=15)

ax_line.set_xlabel("Hidden Layer", fontsize=22, fontweight="bold")
if args.val:
    acc_name = "Validation"
else:
    acc_name = "Training"
ax_line.set_ylabel("{} Accuracy".format(acc_name), fontsize=22, fontweight="bold")

ax_line.legend(artists, ("No Cursor", "Grabbing"), fontsize= args.legend_size, loc = 3)

ax_line.grid(b=True, which='major')

#plt.show()
#plt.show()
if args.save:
    plt.savefig("{}/disc.pdf".format(FIGDIR),bbox_inches='tight',format='pdf',dpi=300)
else:
    plt.show()

halt = True
