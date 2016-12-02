from shapesorting import *

import argparse

import numpy as np
import tensorflow as tf

from shapesorting.autoencoding.autoencoder_lib import *
from sandbox.util import Saver

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--load_dqn',type=bool,default=False)

parser.add_argument('--particular_shape',type=int,default=None)
parser.add_argument('--visualize_data',type=int,default=0)

#parser.add_argument('--exp_name',type=str,default='end')
parser.add_argument('--share_weights',type=bool,default=True)
parser.add_argument('--n_channels',type=int,default=4)

parser.add_argument('--train',type=bool,default=False)

# training
parser.add_argument('--epochs',type=int,default=100)
parser.add_argument('--batch_size',type=int,default=20)

#parser.add_argument('--simi_data_time',type=str,default="16-11-10-01-13AM")
parser.add_argument('--simi_data_time',type=str,default="16-11-10-08-58PM")
parser.add_argument('--encoding_time',type=str,default='16-11-11-07-18PM')

args = parser.parse_args()

assert args.batch_size == 20
    
simidata_saver = Saver(time=args.simi_data_time,path='{}/{}'.format(DATADIR,'simi_data'))
#modelsaver = Saver(path='{}/{}'.format(DATADIR,'classifier'))
modelsaver= None

encoding_saver = Saver(time=args.encoding_time,path='{}/{}'.format(DATADIR,'enco_simi_data'))
#Y = encoding_saver.load_value(0,'Y')
#D = encoding_saver.load_dictionary(0, 'l3_flat_encodings') # more overfitting, more validation accuracy
#D = encoding_saver.load_dictionary(0, 'l2_flat_encodings') 
#D = encoding_saver.load_dictionary(0, 'adv_hid_encodings')
#D = encoding_saver.load_dictionary(0, args.encoding)

simi_data = encoding_saver.load_dictionary(0,'simi_data')
Y = simi_data['Y']
X = simi_data['X1']
SHAPES1 = simi_data['SHAPES1']

N = X.shape[0]
cutoff = int(0.7 * N)

X = np.repeat(X[...,None], 4, axis=-1)

O_t = X[:cutoff]
O_v = X[cutoff:]

X_t = X[:cutoff]
Y_t = SHAPES1[:cutoff]
X_v = X[cutoff:]
Y_v = SHAPES1[cutoff:]

model = Simonyan(5,0, top_layer='adv_hid')

dqnencoder_saver = Saver(time='16-11-11-07-06PM',path='{}/{}'.format(DATADIR,'dqn_weights'))
dqnencoder_weights = dqnencoder_saver.load_dictionary(0,'encoder')

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    model.load_weights(dqnencoder_weights)
    model.train(X_t, Y_t, X_v, Y_v, epochs=10)
    
    p_t, l_t = model.predict(X_t)
    p_v, l_v = model.predict(X_v)
    
    print "Train Acc: {}".format(accuracy_score(Y_t, p_t))
    print "Valid Acc: {}".format(accuracy_score(Y_v, p_v))
    
    model.visprop()    

halt= True