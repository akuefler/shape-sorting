import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from config import DATADIR

from util import Saver

import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--data_time",type=str,default="17-01-22-12-45-05-548393")
parser.add_argument("--linewidth",type=int,default=3)

args = parser.parse_args()

data_saver = Saver(time=args.data_time, path='{}/{}'.format(DATADIR,'disc_results'))
data = data_saver.load_dictionary(0,"data")
MT, MV = data["MT"], data["MV"] # grab conditions x layers x models

MT_mean = MV.mean(axis=-1)
MV_mean = MV.mean(axis=-1)

matplotlib.rcParams.update({'font.size': 22})

## plot both
#f, axs = plt.subplots(1,2)
#for i, color in enumerate(['r','g','b']):
    #axs[0].plot(MT_mean[i,:],color=color,linewidth=args.linewidth)
    #axs[1].plot(MV_mean[i,:],color=color,linewidth=args.linewidth)
    
    #axs[0].set_ylim((0.2,0.8))
    #axs[1].set_ylim((0.2,0.8))
    
f, ax = plt.subplots(figsize=(10,10))
for tick in ax.get_xaxis().get_major_ticks():
    tick.set_pad(20.)
    tick.label1 = tick._get_text1()

for i, color in enumerate(['r','g','b']):
    ax.plot(MV_mean[i,:],color=color,linewidth=args.linewidth)
    
xlo, xhi = ax.get_xlim()

ax.set_xticks(np.arange(xlo, xhi + 1.0))
ax.set_xticklabels(['Conv1','Conv2','Conv3','Val.','Adv.'])

ax.set_xlabel("Hidden Layer", fontsize=28)
ax.set_ylabel("Accuracy", fontsize=28)

plt.show()

halt = True