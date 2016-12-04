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
parser.add_argument('--encoding_time',type=str,default='16-11-11-07-18PM')

args = parser.parse_args()

assert args.batch_size == 20
    
#modelsaver = Saver(path='{}/{}'.format(DATADIR,'classifier'))
modelsaver= None

encoding_saver = Saver(time=args.encoding_time,path='{}/{}'.format(DATADIR,'enco_simi_data'))
#Y = encoding_saver.load_value(0,'Y')
D = encoding_saver.load_dictionary(0, 'l3_flat_encodings') # more overfitting, more validation accuracy
#D = encoding_saver.load_dictionary(0, 'l2_flat_encodings') 
#D = encoding_saver.load_dictionary(0, 'adv_hid_encodings')
#D = encoding_saver.load_dictionary(0, args.encoding)

simi_data = encoding_saver.load_dictionary(0,'simi_data')
Y = simi_data['Y']
X = simi_data['X1']
SHAPES = simi_data['SHAPES1']

N = X.shape[0]
cutoff = int(0.7 * N)

p = np.random.permutation(range(N))
#X = X[p]
#SHAPES = SHAPES[p]
X = np.repeat(X[...,None], 4, axis=-1)

X_t = X[:cutoff]
Y_t = SHAPES[:cutoff]
X_v = X[cutoff:]
Y_v = SHAPES[cutoff:]

model = Simonyan(5,0, top_layer='adv_hid', dueling=True)

# no weights: 46, 37

#dqnencoder_saver = Saver(time='16-11-11-07-04PM',path='{}/{}'.format(DATADIR,'dqn_weights')) # t:30,v:26
dqnencoder_saver = Saver(time='16-11-11-07-06PM',path='{}/{}'.format(DATADIR,'dqn_weights')) # t:37, v:30

dqnencoder_weights = dqnencoder_saver.load_dictionary(0,'encoder')

#np.random.seed(456)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    model.load_weights(dqnencoder_weights)
    #model.train(X_t, Y_t, X_v, Y_v, epochs=200)
    
    #import h5py
    #with h5py.File('softmax_weights.h5','r') as hf:
        #w = hf['w'][...]
        #b = hf['b'][...]
        
    Z_t = model.encode(X_t, layer='adv_hid')
    Z_v = model.encode(X_v, layer='adv_hid')
    
    from sklearn.linear_model import LogisticRegression
    sk_model = LogisticRegression(solver='sag',multi_class='multinomial')
    sk_model.fit(Z_t, Y_t)
    
    ops = [model.cls_w.assign(sk_model.coef_.T),model.cls_b.assign(sk_model.intercept_)]
    sess.run(ops)

    skp_t = sk_model.predict(Z_t)
    skp_v = sk_model.predict(Z_v)
    
    
    print "Logistic: "
    print "Train Acc: {}".format(accuracy_score(Y_t, skp_t))
    print "Valid Acc: {}".format(accuracy_score(Y_v, skp_v))    
    
    p_t, l_t = model.predict(X_t)
    p_v, l_v = model.predict(X_v)
    
    print "Entire Model: "
    print "Train Acc: {}".format(accuracy_score(Y_t, p_t))
    print "Valid Acc: {}".format(accuracy_score(Y_v, p_v))
    
    ## load weights, 50 epochs
    # 35.67 train, 32.43 valid
    ## load weights, 100 epochs
    # 38.17 train, 32.22 valid
    
    I = model.visprop()    

halt= True