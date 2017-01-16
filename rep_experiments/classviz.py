from shapesorting import *

import argparse

import numpy as np
import tensorflow as tf

from shapesorting.autoencoding.autoencoder_lib import *
from sandbox.util import Saver

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

parser = argparse.ArgumentParser()

parser.add_argument('--load_dqn',type=bool,default=False)

parser.add_argument('--particular_shape',type=int,default=None)
parser.add_argument('--visualize_data',type=int,default=0)

#parser.add_argument('--exp_name',type=str,default='end')
parser.add_argument('--share_weights',type=bool,default=True)
parser.add_argument('--n_channels',type=int,default=4)

parser.add_argument('--train',type=bool,default=False)

# training
parser.add_argument('--epochs',type=int,default=10000)
parser.add_argument('--batch_size',type=int,default=20)

#parser.add_argument('--simi_data_time',type=str,default="16-11-10-01-13AM")
parser.add_argument('--encoding_time',type=str,default='0')
parser.add_argument('--encoding_layer',type=str,default='l2_flat')

args = parser.parse_args()

assert args.batch_size == 20
    
#modelsaver = Saver(path='{}/{}'.format(DATADIR,'classifier'))
modelsaver= None

encoding_saver = Saver(time=args.encoding_time,path='{}/{}'.format(DATADIR,'enco_simi_data'))
#Y = encoding_saver.load_value(0,'Y')
#D = encoding_saver.load_dictionary(0, 'l3_flat_encodings') # more overfitting, more validation accuracy
#D = encoding_saver.load_dictionary(0, 'l2_flat_encodings') 
#D = encoding_saver.load_dictionary(0, 'adv_hid_encodings')
#D = encoding_saver.load_dictionary(0, args.encoding)

simi_data = encoding_saver.load_dictionary(0,'simi_data')
X = simi_data['X1']
SHAPES = simi_data['SHAPES1']

N = X.shape[0]
N = 50
cutoff = int(0.7 * N)

p = np.random.permutation(range(N))
X = X[p]
SHAPES = SHAPES[p]
X = np.repeat(X[...,None], 4, axis=-1)

X_t = X[:cutoff]
Y_t = SHAPES[:cutoff]
X_v = X[cutoff:]
Y_v = SHAPES[cutoff:]

# [Trapezoid, RightTri, Hexagon, Tri, Rect]
def initializer(shape, dtype=None, partition_info=None):
    return np.random.normal(X_t.mean(axis=0), X_t.std(axis=0) + 1e-8)[None,...]

# no weights: 46, 37

#dqnencoder_saver = Saver(time='16-11-11-07-04PM',path='{}/{}'.format(DATADIR,'dqn_weights')) # t:30,v:26
dqnencoder_saver = Saver(time='16-11-11-07-06PM',path='{}/{}'.format(DATADIR,'dqn_weights')) # t:37, v:30

dqnencoder_weights = dqnencoder_saver.load_dictionary(0,'encoder')

#np.random.seed(456)

#f, axs = plt.subplots(5,4)
#for i in range(5):
    #for j in range(4):
        #axs[i,j].imshow(X_t[Y_t==i].mean(axis=0)[:,:,j])
        #axs[i,j].imshow(X_t[Y_t==i].mean(axis=0)[:,:,j])
        
halt= True

model = Simonyan(5,top_layer=args.encoding_layer, dueling=True, reg=0.01, I_initializer= initializer)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    model.load_weights(dqnencoder_weights)
    #model.train(X_t, Y_t, X_v, Y_v, epochs=200)
    
    Z_t = model.encode(X_t, layer=args.encoding_layer)
    Z_v = model.encode(X_v, layer=args.encoding_layer)    
    
    import h5py
    if args.train:
        sk_model = LogisticRegression(solver='sag',multi_class='multinomial')
        #sk_model = SVC(kernel='linear')
        #sk_model = LinearSVC(dual=False, C=0.1)
        sk_model.fit(Z_t, Y_t)
        
        skp_t = sk_model.predict(Z_t)
        skp_v = sk_model.predict(Z_v)
        
        print "Logistic: "
        print "Train Acc: {}".format(accuracy_score(Y_t, skp_t))
        print "Valid Acc: {}".format(accuracy_score(Y_v, skp_v))
        
        with h5py.File('lr_weights_grab{}_{}.h5'.format(args.encoding_time,args.encoding_layer),'a') as hf:
            #w = hf['w'][...]
            #b = hf['b'][...]
            hf.create_dataset("w",data=sk_model.coef_.T)
            hf.create_dataset("b",data=sk_model.intercept_)        
    
        assert False
    with h5py.File('lr_weights_grab{}_{}.h5'.format(args.encoding_time,
                                                    args.encoding_layer),'r') as hf:
        w = hf['w'][...]
        b = hf['b'][...]
    
    #ops = [model.cls_w.assign(sk_model.coef_.T),model.cls_b.assign(sk_model.intercept_)]
    ops = [model.cls_w.assign(w),model.cls_b.assign(b)]    
    sess.run(ops)

    #skp_t = sk_model.predict(Z_t)
    #skp_v = sk_model.predict(Z_v)
    
    #print "Logistic: "
    #print "Train Acc: {}".format(accuracy_score(Y_t, skp_t))
    #print "Valid Acc: {}".format(accuracy_score(Y_v, skp_v))
    
    #with h5py.File('lr_weights.h5','a') as hf:
        ##w = hf['w'][...]
        ##b = hf['b'][...]
        #hf.create_dataset("w",data=sk_model.coef_.T)
        #hf.create_dataset("b",data=sk_model.intercept_)
    
    p_t, l_t = model.predict(X_t)
    p_v, l_v = model.predict(X_v)
    
    print "Entire Model: "
    print "Train Acc: {}".format(accuracy_score(Y_t, p_t))
    print "Valid Acc: {}".format(accuracy_score(Y_v, p_v))
    
    Is = []
    for i in range(5):
        #I = model.visprop(cls_ix=i,epochs=1)
        I = model.visprop(cls_ix=i,epochs=args.epochs)
        
        p_i, l_i = model.predict(I)
        print "Predicted for I: {}".format(p_i)
        Is.append(I)

f, axs = plt.subplots(5,4)
plt.tight_layout()
axs
labels = ["Trapezoid", "Right Tri.", "Hexagon", "Equil. Tri.", "Square"]
for i, axrow in enumerate(axs):
    I = Is[i]
    for j, ax in enumerate(axrow):
        if j == 0:
            ax.set_ylabel(labels[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        img = I[0,:,:,j]
        ax.imshow(img,cmap="Greys")
        
plt.tight_layout()
        
#for i in range(4):
    #img = I[0,:,:,i]
    ##img = (img - img.min())/(img.max() - img.min())
    #axs[0,i].imshow(img,cmap="jet")   
    ##axs[1,i].imshow(I[0,:,:,i] + X_t.mean(axis=0)[:,:,i])
    #axs[1,i].imshow(np.concatenate([I[0,:,:,i],X_t[Y_t==i].mean(axis=0)[:,:,i]],axis=1),cmap="jet")
    
plt.show()

halt= True