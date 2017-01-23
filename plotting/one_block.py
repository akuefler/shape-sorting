import argparse
import numpy as np

import matplotlib.pyplot as plt

from config import DATADIR

from util import Saver

import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--data_time",type=str,default="17-01-22-16-00-14-471258")

parser.add_argument("--n_segments",type=int,default=4)
parser.add_argument("--normalize",type=int,default=1)

parser.add_argument("--plot_by_action",type=int,default=0)

parser.add_argument("--bar_width",type=float,default=0.25)
parser.add_argument("--legend_size",type=float,default=25.)

args = parser.parse_args()

data_saver = Saver(time=args.data_time, path='{}/{}'.format(DATADIR,'one_block_results'))

actions_after_grab = data_saver.load_recursive_dictionary("actions_after_grab")["actions_after_grab"]
stats = data_saver.load_dictionary(0,"stats")["stats"]

afg = [np.array(x) for x in actions_after_grab.values()]

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
    for j, color in enumerate(['r','g','b']):
        artist = ax[i].bar(ind + j * args.bar_width, M_shape[j], args.bar_width, color=color)
        artists.append(artist[0])
        
    ax[i].set_ylim(0,ylim)
    ax[i].legend(tuple(artists), ("Grab","Move","Rotate"), fontsize= args.legend_size)
        
        
# plot "optimality"
    
plt.show()

halt= True