import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from config import *
from plotting import *

from util import Saver

import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--data_time",type=str,default="17-01-30-21-13-16-702347")

parser.add_argument("--n_segments",type=int,default=4)
parser.add_argument("--normalize",type=int,default=1)

parser.add_argument("--plot_by_action",type=int,default=0)

parser.add_argument("--bar_width",type=float,default=0.3)
parser.add_argument("--legend_size",type=float,default=25.)

args = parser.parse_args()

data_saver = Saver(time=args.data_time, path='{}/{}'.format(DATADIR,'one_block_results'))

actions_after_grab = data_saver.load_recursive_dictionary("actions_after_grab")["actions_after_grab"]
stats = data_saver.load_dictionary(0,"stats")["stats"]

matplotlib.rcParams.update({'font.size': 22})

afg = [np.array(x) for x in actions_after_grab.values()]
# remove wrong fits
print("{} wrong fits".format(stats[:,-1].sum()))
print("{} ave. wrong fits".format(stats[:,-1].mean()))

stats = stats[stats[:,-1] == 0]
stats = stats[:,:-1]

# action x segment
if args.plot_by_action:
    n_axes = 5
else:
    n_axes = 1

ind = np.arange(args.n_segments)
M = np.zeros((n_axes,3,args.n_segments))

for i, a in enumerate(afg):
    if args.plot_by_action:
        ix = stats[i,0]
    else:
        ix = 0    

    #beg, mid, end = np.array_split(a,3)
    a = a[:-1] # remove last "grab" action
    for j, arr in enumerate(np.array_split(a,args.n_segments)):
        for k, act in enumerate(arr):
            if act != -1:
                M[ix, act,j] += 1
            
if args.normalize:
    assert not args.plot_by_action
    M = np.squeeze(M)
    m = M.sum(axis=0)
    M /= m
    M = M[None,...]
    
    ylim = 1.0
else:
    ylim = M.max()

# plot "strategy"            
f, ax = plt.subplots(1,n_axes, figsize=(n_axes * 10,10))
if n_axes == 1:
    ax = np.array([ax])
for i, M_shape in enumerate(M):
    artists = []
    for j, color in enumerate([(.5,1.,.5),(1,.5,.5),(.5,.5,1)]):
        artist = ax[i].bar(ind + j * args.bar_width, M_shape[j], args.bar_width, color=color,
                           linewidth=3)
        artists.append(artist[0])
        
    ax[i].set_ylim(0,ylim)
    ax[i].legend(tuple(artists), ("Grab","Move","Rotate"), fontsize= args.legend_size)
        
    ax[i].set_xticks(range(5))
    ax[i].grid(b=True, which='major')
    
    ax[i].set_xlabel("Trajectory Segment", fontsize=22, fontweight="bold")
    ax[i].set_ylabel("Action Distribution", fontsize=22, fontweight="bold")    

plt.savefig("{}strategy.pdf".format(FIGDIR),bbox_inches='tight',format='pdf',dpi=300)

names = np.array(["Trap.", "R. Tri", "Hex.", "E. Tri", "Square"])

l = []
for i in range(5):
    x = stats[stats[:,0] == i][:,1:]
    d = {"name":names[i], "mean":x.mean(axis=0), "median":np.median(x,axis=0), "std":x.std(axis=0)}
    l.append(
        d
    )
    
print("Shape & Min. Steps &  Act. Steps & Ratio \\\\")
print("\hline")
for i in SHAPE_ORDER:
    ll = l[i]
    min_steps_mu = ll['mean'][0]
    steps_taken_mu = ll['mean'][1]
    min_steps_std = ll['std'][0]
    steps_taken_std = ll['std'][1]    
    print("{} & {:.2f} \pm ({:.1f}) & {:.2f} \pm ({:.1f}) & {:.2f} \\\\".format(ll['name'], min_steps_mu, min_steps_std,
                                                             steps_taken_mu, steps_taken_std,
                                                             min_steps_mu/steps_taken_mu,
                                                             ))

halt= True