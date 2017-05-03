import numpy as np
import re

from config import *
from plotting import *

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 22})

fin = open("oct28_dqn_log.txt",'r')
X = []
itr = []
while True:
    s = fin.readline()
    if 'avg_r' in s:
        halt= True
        sp = ',' + s
        keys= re.findall("\,(.*?)\:",sp)
        vals= re.findall("\:(.*?)[\,\n]",s)
        assert len(keys) == len(vals)
        X.append(
            np.array(vals,dtype=float)
        )
    if s == '':
        X = np.array(X)
        break
    if "%" in s:
        itr.append(
        s.split('/')[0].split('|')[-1]
        )

#d_keys = [' avg_ep_r',' # game']
d_keys = [' # game']
titles = ["Average Return per Trial", "Number of Trials"]
f, axs = plt.subplots(len(d_keys),1, figsize=(15,15))
axs = [axs]

for j, (d_key, ax) in enumerate(zip(d_keys,axs)):
    ax.set_xlim((0,1800))
    #ax.set_xticklabels([])
    #ax.set_xticks([])
    #if d_key != ' # game':
        #ax.set_yticklabels([])
        #ax.set_yticks([])

    i = keys.index(d_key)
    ax.plot(X[:,i], linewidth=3, color=[.5,.5,1.])
    #ax.set_title(titles[j])
    ax.set_ylabel(titles[j], fontweight="bold")
    ax.grid(b=True, which='major')
    if j == len(d_keys) - 1:
        ax.set_xlabel("Epochs", fontweight="bold")

plt.tight_layout()

if True:
    plt.savefig("{}/dqn_epochs.pdf".format(FIGDIR),bbox_inches='tight',format='pdf',dpi=300)
else:
    plt.show()

