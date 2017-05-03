import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from config import *
from plotting import *

from util import Saver

import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--data_time",type=str,default="17-01-29-09-48-32-552854")

args = parser.parse_args()

data_saver = Saver(time=args.data_time, path='{}/{}'.format(DATADIR,'one_block_results'))
stats = data_saver.load_dictionary(0,"stats")["stats"]

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
