import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import random

from sklearn.metrics import accuracy_score

n_class = 5
n_feat = 512
n_example = 100000

factor = 5.0

C1 = np.random.randint(0,n_class,(n_example,))
C2 = np.random.randint(0,n_class,(n_example,))

X1 = np.eye(n_class)[C1] * factor + np.random.normal(0,.6,(n_example,n_class))
X2 = np.eye(n_class)[C2] * factor + np.random.normal(0,.6,(n_example,n_class))

np.random.seed(456)
F1 = np.random.randn(n_class,n_feat)
F2 = np.random.randn(n_feat,n_feat)

X1 = np.matmul(np.tanh(np.matmul(X1,F1)),F2)
X2 = np.matmul(np.tanh(np.matmul(X2,F1)),F2)

X = np.column_stack([X1,X2])
C = (C1 == C2).astype("int32")

keep_ix = np.logical_not(np.logical_and((C==0),np.random.uniform(0,1,C.shape) > 1.0/n_class))
X = X[keep_ix]
C = C[keep_ix]

n_example, n_feat = X.shape

X_t = X[:int(0.7 * n_example)]
Y_t = C[:int(0.7 * n_example)]
X_v = X[int(0.7 * n_example):]
Y_v = C[int(0.7 * n_example):]

N_t = len(Y_t)
N_v = len(Y_v)

I = n_feat
H = 256
O = 2

batch_size = 10
epochs = 10

fn = tf.nn.relu

x = tf.placeholder(tf.float32, shape=(None,I))
y = tf.placeholder(tf.int32, shape=(None,))

W1 = tf.get_variable("W1",shape=(I,H),dtype=tf.float32)
b1 = tf.get_variable("b1",shape=(H,),dtype=tf.float32)

W2 = tf.get_variable("W2",shape=(H,O),dtype=tf.float32)
b2 = tf.get_variable("b2",shape=(O,),dtype=tf.float32)

logits = tf.nn.xw_plus_b(fn(tf.nn.xw_plus_b(x, W1, b1)), W2, b2)

obj = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
opt = tf.train.AdamOptimizer().minimize(obj)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    for epoch in xrange(epochs):
        minibatches = range(0,N_t,batch_size)
        random.shuffle(minibatches)
        losses = []
        for i in minibatches:
            X_t_batch = X_t[i:i+batch_size]
            Y_t_batch = Y_t[i:i+batch_size]
            
            loss, _ = sess.run([obj,opt],{x:X_t_batch,y:Y_t_batch})
            losses.append(loss)
            
        print("Epoch {}: {}".format(epoch,np.mean(losses)))
        
    p_t = sess.run(tf.argmax(logits,1),{x:X_t})
    p_v = sess.run(tf.argmax(logits,1),{x:X_v})
        
    print("Training Accuracy: {}".format(accuracy_score(Y_t, p_t)))
    print("Validation Accuracy: {}".format(accuracy_score(Y_v, p_v)))
            
halt= True