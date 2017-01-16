import numpy as np
import argparse
import h5py

import matplotlib.pyplot as plt

from shapesorting import *

#parser = argparse.ArgumentParser()

#parser.add_argument('--load_dqn',type=int,default=False)

#args = parser.parse_args()

#modelsaver = Saver(path='{}/{}'.format(DATADIR,'classifier'))

#modelsaver.load

halt= True

t_loss, v_loss = [], []
with h5py.File('../data/classifier/16-12-13-12-12AM/epochs.h5','r') as hf:
    for itr, d in hf.iteritems():
        t_loss.append(d['loss_t'][...])
        v_loss.append(d['loss_v'][...])
        
f, ax = plt.subplots(1,1)
ax.plot(t_loss,'b')
ax.plot(v_loss,'r')
ax.set_title("End-to-End Learning")

ax.set_xlim([0,24])
        
plt.show()
halt= True