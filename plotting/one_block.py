import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from config import *
from plotting import *

from util import Saver

import tqdm

parser = argparse.ArgumentParser()

#parser.add_argument("--data_time",type=str,default="17-01-30-21-13-16-702347") # original results

parser.add_argument("--n_segments",type=int,default=4)
parser.add_argument("--normalize",type=int,default=1)
parser.add_argument("--plot_by_action",type=int,default=0)
parser.add_argument("--bar_width",type=float,default=0.3)
parser.add_argument("--legend_size",type=float,default=25.)

#parser.add_argument("--data_time",type=str,default="17-04-22-18-08-15-962612") # test_ep is 0.0
parser.add_argument("--data_time",type=str,default="17-04-22-20-24-49-443199") # test_ep is 0.1

args = parser.parse_args()

data_saver = Saver(time=args.data_time, path='{}/{}'.format(DATADIR,'one_block_results'))

actions_after_grab = data_saver.load_recursive_dictionary("actions_after_grab")["actions_after_grab"]
stats_headers = \
    ["winners","step_min","step_taken","wrong_fits","first_vs","contact_vs"]
stats = data_saver.load_dictionary(0,"stats")["stats"] # numpy array

try:
    exp_args = data_saver.load_dictionary(0,"args")
except:
    exp_args = None

matplotlib.rcParams.update({'font.size': 22})

afg = [np.array(x) for x in actions_after_grab.values()]
# remove wrong fits
print("{} wrong fits".format(stats[:,stats_headers.index("wrong_fits")].sum()))
print("{} ave. wrong fits".format(stats[:,stats_headers.index("wrong_fits")].mean()))

if exp_args is not None:
    print("Number of time outs: {}".format(exp_args["n_episodes"] - stats.shape[0]))

stats = stats[stats[:,stats_headers.index("wrong_fits")] == 0]
#stats = stats[:,:stats_headers.index("wrong_fits")]

# action x segment
if args.plot_by_action:
    n_axes = 5
else:
    n_axes = 1

ind = np.arange(args.n_segments)
M = np.zeros((n_axes,3,args.n_segments))

for i, a in enumerate(afg):
    if args.plot_by_action:
        ix = stats[i,stats_headers.index("winners")]
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

plt.savefig("{}/strategy.pdf".format(FIGDIR),bbox_inches='tight',format='pdf',dpi=300)

l = []
for i in range(5):
    x = stats[stats[:,stats_headers.index("winners")] == i] #[:,1:]
    d = {"name":LABELS[i], "mean":x.mean(axis=0), "median":np.median(x,axis=0), "std":x.std(axis=0)}
    l.append(d)

if exp_args is not None:
    """
    revised experiments, now tracking values and storing parameters like the
    number of trials and the test epsilon.
    """
    print("retained data: {}".format(stats.shape[0] / float(exp_args["n_episodes"])))
    print("Shape & Value & Min. Steps &  Act. Steps & Ratio \\\\")
    print("\hline")
    for i in SHAPE_ORDER:
        ll = l[i]
        min_steps_mu = ll['mean'][stats_headers.index("step_min")]
        steps_taken_mu = ll['mean'][stats_headers.index("step_taken")]
        min_steps_std = ll['std'][stats_headers.index("step_min")]
        steps_taken_std = ll['std'][stats_headers.index("step_taken")]

        #first_v = ll['mean'][stats_headers.index("first_vs")]
        contact_v = ll['mean'][stats_headers.index("contact_vs")]
        print("{} & {:.2f} & {:.2f} \pm ({:.1f}) & {:.2f} \pm ({:.1f}) & {:.2f}\\\\".format(ll['name'],
                                                                 contact_v,
                                                                 min_steps_mu, min_steps_std,
                                                                 steps_taken_mu, steps_taken_std,
                                                                 min_steps_mu/steps_taken_mu
                                                                 ))
else: # results from original manuscript
    import pdb; pdb.set_trace()
    print("retained data: {}".format(stats.shape[0] / 50000.))
    print("Shape & Min. Steps &  Act. Steps & Ratio & Value \\\\")
    print("\hline")
    for i in SHAPE_ORDER:
        ll = l[i]
        min_steps_mu = ll['mean'][stats_headers.index("step_min")]
        steps_taken_mu = ll['mean'][stats_headers.index("step_taken")]
        min_steps_std = ll['std'][stats_headers.index("step_min")]
        steps_taken_std = ll['std'][stats_headers.index("step_taken")]

        print("{} & {:.2f} \pm ({:.1f}) & {:.2f} \pm ({:.1f}) & {:.2f} \\\\".format(ll['name'],
                                                                 min_steps_mu, min_steps_std,
                                                                 steps_taken_mu, steps_taken_std,
                                                                 min_steps_mu/steps_taken_mu,
                                                                 ))

halt= True
