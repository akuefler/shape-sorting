import argparse
import numpy as np

from plotting import *

import matplotlib
import matplotlib.pyplot as plt

from config import *

from util import Saver

import itertools
import collections

parser = argparse.ArgumentParser()

parser.add_argument("--data_time",type=str,default="17-01-28-13-58-54-152302")
parser.add_argument("--linewidth",type=int,default=5)
parser.add_argument("--legend_size",type=int,default=25)
parser.add_argument("--val",type=int,default=1)

args = parser.parse_args()

data_saver = Saver(time=args.data_time, path='{}/{}'.format(DATADIR,'disc_results'))
data = data_saver.load_dictionary(0,"data")
MT, MV = data["MT"], data["MV"] # grab conditions x layers x models

MT_mean = MV.mean(axis=-1)
MV_mean = MV.mean(axis=-1)
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
    C = data['CV'][0,best_layer].mean(axis=0)
    C_grab = data['CV'][1,best_layer].mean(axis=0)
    
    C, c_argsort = argsort_matrix(C)
    C_grab, c_argsort_ = argsort_matrix(C_grab)
    
    assert np.all(c_argsort == c_argsort_)
    
else:
    M = MT_mean
    C = data['CT'][0,best_layer].mean(axis=0)
    C_grab = data['CT'][1,best_layer].mean(axis=0)
    
    C, c_argsort = argsort_matrix(C)
  
f, ax = plt.subplots(figsize=(10,10))
for tick in ax.get_xaxis().get_major_ticks():
    tick.set_pad(20.)
    tick.label1 = tick._get_text1()

artists = []
for i, color in enumerate(['r','b']):
    artist = ax.plot(M[i,:],color=color,linewidth=args.linewidth)
    artists.append(artist[0])
    
xlo, xhi = ax.get_xlim()

ax.set_xticks(np.arange(xlo, xhi + 1.0))
ax.set_xticklabels(['Conv1','Conv2','Conv3','Val.','Adv.'])

ax.set_xlabel("Hidden Layer", fontsize=22, fontweight="bold")
ax.set_ylabel("Accuracy", fontsize=22, fontweight="bold")

ax.legend(artists, ("No Cursor", "Grabbing"), fontsize= args.legend_size, loc = 3)

ax.grid(b=True, which='major')

plt.savefig("{}disc_acc.png".format(FIGDIR),bbox_inches='tight')
plt.cla()
plot_matrix([C_grab], range(5), LABELS[c_argsort], names=[None], cmap=plt.cm.Blues, normalize=True)
plt.savefig("{}conf_mat.png".format(FIGDIR),bbox_inches='tight')

halt = True