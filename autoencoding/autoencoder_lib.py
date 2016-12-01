import numpy as np
import tensorflow as tf
from game import ShapeSorter
from game_settings import SHAPESORT_ARGS
import keras as ks
import keras.backend as K
import keras.layers as KL

from dqn_master.dqn.ops import linear, conv2d

import random

from game_settings import DISCRETE_ACT_MAP4
from rllab.sampler.utils import rollout

#assert K.backend() == 'tensorflow'

from sandbox.util import Saver

env = ShapeSorter(**SHAPESORT_ARGS[1])

input_shape= (84,84,4)  # TENSORFLOW INPUT SHAPE

activation_fn = tf.nn.relu

#bias = K.variable(np.zeros((output_shape[-1],)))
#model.add(KL.Lambda(lambda x : x + bias, output_shape= output_shape[1:]))

class Model():
    def predict(self, X):
        if X.shape[0] == self.batch_size:
            pred, logit = self._predict(X)
        elif X.shape[0] % self.batch_size == 0:
            preds = []
            logits = []
            for i in range(0,X.shape[0],self.batch_size):
                p, l = self._predict(X[i:self.batch_size+i])
                preds.append(p)
                logits.append(l)
            pred = np.concatenate(preds)
            logit = np.concatenate(logits)
                
        else:
            raise NotImplementedError
        return pred, logit
            
    def _predict(self, X):
        sess = tf.get_default_session()
        if self.supervised:
            X1, X2 = np.split(X,2,axis=1) 
            feed = {self.x: np.squeeze(X1), self.xp: np.squeeze(X2)}
        else:
            feed = {self.x: X}        

        X_hat, LOGITS = sess.run([self.x_hat,self.logits],feed)
        return X_hat, LOGITS

class AutoEncoder(Model):
    def __init__(self, batch_size, dueling= True, supervised=False, trainable=False, top_layer='l3_flat'):
        
        self.batch_size = batch_size
        
        self.layers = {}
        self.encoder_weights = {}
        self.decoder_weights = {}
        self.classifier_weights = {}
        self.dueling = dueling
        self.supervised = supervised
        
        self.x = x = tf.placeholder(tf.float32, shape=(batch_size,84,84,4))
        if supervised:
            self.y = tf.placeholder(tf.int32, shape=(batch_size,))
        else:
            self.y = tf.placeholder(tf.float32, shape=(batch_size,84,84,4))
            
        with tf.variable_scope('encoder') as scope:
            z1 = self.encoder(x, top_layer= top_layer)
            if supervised:
                scope.reuse_variables()
                self.xp = xp = tf.placeholder(tf.float32, shape=(batch_size,84,84,4))                
                z2 = self.encoder(xp, top_layer= top_layer)
                    
                #z = tf.concat(1,[z1,z2])
                z = z1 * z2
        
        self.layer_names = ['l1','l2','l3','l4','l5','q']
        
        if supervised:
            self.classifier(z)
        else:
            self.decoder(layer=top_layer)
            
        var_list = self.classifier_weights.values()
        if trainable:
            var_list += self.encoder_weights.values()
            
        self.opt = tf.train.AdamOptimizer().minimize(self.cost, var_list= var_list)
        
        
    def classifier(self, x, n_layers= 1, n_units = 128):
        
        #self.class_hid1, self.classifier_weights['class_hid1_w'], self.classifier_weights['class_hid1_b'] = \
            #linear(x, n_units, activation_fn=activation_fn, name='class_hid1')
        #self.class_hid2, self.classifier_weights['class_hid2_w'], self.classifier_weights['class_hid2_b'] = \
            #linear(self.class_hid1, n_units, activation_fn=activation_fn, name='class_hid2')
        h = x
        for i in range(n_layers):
            h, self.classifier_weights['class_hid1_w'], self.classifier_weights['class_hid1_b'] = \
                linear(h, n_units, activation_fn=activation_fn, name='class_hid{}'.format(i))            

        self.logits, self.classifier_weights['output_w'], self.classifier_weights['output_b'] = \
            linear(h, 2, name='logits')            
        
        self.x_hat = tf.argmax(self.logits,1)
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.y)
        )
               
        
    def encoder(self, x, top_layer = 'l3_flat'):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        
        self.l1, self.encoder_weights['l1_w'], self.encoder_weights['l1_b'] = conv2d(x,
                                                             32, [8, 8], [4, 4], initializer, activation_fn, 'NHWC', name='l1')
        self.l2, self.encoder_weights['l2_w'], self.encoder_weights['l2_b'] = conv2d(self.l1,
                                                         64, [4, 4], [2, 2], initializer, activation_fn, 'NHWC', name='l2')
        self.l3, self.encoder_weights['l3_w'], self.encoder_weights['l3_b'] = conv2d(self.l2,
                                                         64, [3, 3], [1, 1], initializer, activation_fn, 'NHWC', name='l3')
    
        self.l1_flat = tf.reshape(self.l1, [-1, reduce(lambda x, y: x * y, self.l1.get_shape().as_list()[1:])])
        self.l2_flat = tf.reshape(self.l2, [-1, reduce(lambda x, y: x * y, self.l2.get_shape().as_list()[1:])])
        self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, self.l3.get_shape().as_list()[1:])])
        
        self.layers['l1_flat'] = self.l1_flat
        self.layers['l2_flat'] = self.l2_flat
        self.layers['l3_flat'] = self.l3_flat
        
        if self.dueling:
            self.value_hid, self.encoder_weights['l4_val_w'], self.encoder_weights['l4_val_b'] = \
                linear(self.l3_flat, 512, activation_fn=activation_fn, name='value_hid')
        
            self.adv_hid, self.encoder_weights['l4_adv_w'], self.encoder_weights['l4_adv_b'] = \
                linear(self.l3_flat, 512, activation_fn=activation_fn, name='adv_hid')
        
            self.value, self.encoder_weights['val_w_out'], self.encoder_weights['val_w_b'] = \
                linear(self.value_hid, 1, name='value_out')
        
            self.advantage, self.encoder_weights['adv_w_out'], self.encoder_weights['adv_w_b'] = \
                linear(self.adv_hid, 7, name='adv_out')
            
            self.layers['value_hid'] = self.value_hid
            self.layers['adv_hid'] = self.adv_hid
            self.layers['value'] = self.value
            self.layers['advtantage'] = self.advantage
        
            # Average Dueling
            self.q = self.value + (self.advantage - 
                                   tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
            
            self.decoder_weights['l4_w_t'] = l4_w_t = tf.transpose(self.encoder_weights['l4_adv_w']) 
            self.decoder_weights['q_w_t'] = q_w_t = tf.transpose(self.encoder_weights['adv_w_out'])
            
            l4_w = self.encoder_weights['l4_adv_w']
            q_w = self.encoder_weights['adv_w_out']
            
        else:
            self.l4, self.encoder_weights['l4_w'], self.encoder_weights['l4_b'] = \
                linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
            self.q, self.encoder_weights['q_w'], self.encoder_weights['q_b'] = \
                linear(self.l4, 7, name='q')            
            
            self.layers['l4'] = self.l4
            
            self.decoder_weights['l4_w_t'] = l4_w_t = tf.transpose(self.encoder_weights['l4_w'])                
            self.decoder_weights['q_w_t'] = q_w_t = tf.transpose(self.encoder_weights['q_w'])
            
            l4_w = self.encoder_weights['l4_w']
            q_w = self.encoder_weights['q_w']
        
        self.layers['l1'] = self.l1
        self.layers['l2'] = self.l2
        self.layers['l3'] = self.l3
        self.layers['q'] = self.q
        
        return self.layers[top_layer]
            
    def decoder(self, layer):
        if True:
            return None
        output_shapes = [(20, 9, 9, 64), (20, 20, 20, 32), (20, 84, 84, 4)]
        
        #self.decoder_weights['q_w_t'] = q_w_t = tf.transpose(self.encoder_weights['q_w'])
        #self.decoder_weights['l4_w_t'] = l4_w_t = tf.transpose(self.encoder_weights['l4_w'])
        self.decoder_weights['l3_w_t'] = l3_w_t = tf.transpose(self.encoder_weights['l3_w'],[0,1,2,3])
        self.decoder_weights['l2_w_t'] = l2_w_t = tf.transpose(self.encoder_weights['l2_w'],[0,1,2,3])
        self.decoder_weights['1l_w_t'] = l1_w_t = tf.transpose(self.encoder_weights['l1_w'],[0,1,2,3])
        
        self.decoder_weights['q_b_t'] = q_b_t = tf.Variable(np.zeros(q_w_t.get_shape()[-1].value,).astype('float32'))
        self.decoder_weights['l4_b_t'] = l4_b_t = tf.Variable(np.zeros(64,).astype('float32')) # derp.
        self.decoder_weights['l3_b_t'] = l3_b_t = tf.Variable(np.zeros(l3_w_t.get_shape()[-2].value,).astype('float32'))
        self.decoder_weights['l2_b_t'] = l2_b_t = tf.Variable(np.zeros(l2_w_t.get_shape()[-2].value,).astype('float32'))
        self.decoder_weights['l1_b_t'] = l1_b_t = tf.Variable(np.zeros(l1_w_t.get_shape()[-2].value,).astype('float32'))
        
        scope.reuse_variables()
        self.q_T = activation_fn(tf.nn.xw_plus_b(self.q, tf.transpose(q_w), q_b_t, name='q_T'))

        self.l4_T = tf.matmul(self.q_T, tf.transpose(l4_w))
        self.l4_T_unflat = activation_fn( tf.reshape(self.l4_T, self.l3.get_shape()) + l4_b_t)
        
        self.l3_T = activation_fn(tf.nn.conv2d_transpose(self.l4_T_unflat, l3_w_t, output_shapes[0], 
                                          [1,1,1,1],padding="VALID",name='l3_T') + l3_b_t)
        self.l2_T = activation_fn(tf.nn.conv2d_transpose(self.l3_T, l2_w_t, output_shapes[1], 
                                                      [1,2,2,1],padding="VALID",name='l3_T') + l2_b_t)
        self.l1_T = tf.nn.conv2d_transpose(self.l2_T, l1_w_t, output_shapes[2], 
                                                      [1,4,4,1],padding="VALID",name='l1_T')+ l1_b_t
        
        self.x_hat = self.l1_T
        self.cost = tf.reduce_mean(
            tf.nn.l2_loss(self.x_hat - self.x)
            )
        
    #def predict(self, X):
        #if X.shape[0] == self.batch_size:
            #pred, logit = self._predict(X)
        #elif X.shape[0] % self.batch_size == 0:
            #preds = []
            #logits = []
            #for i in range(0,X.shape[0],self.batch_size):
                #p, l = self._predict(X[i:self.batch_size+i])
                #preds.append(p)
                #logits.append(l)
            #pred = np.concatenate(preds)
            #logit = np.concatenate(logits)
                
        #else:
            #raise NotImplementedError
        #return pred, logit
            
            
            
    #def _predict(self, X):
        #sess = tf.get_default_session()
        #if self.supervised:
            #X1, X2 = np.split(X,2,axis=1) 
            #feed = {self.x: np.squeeze(X1), self.xp: np.squeeze(X2)}
        #else:
            #feed = {self.x: X}        

        #X_hat, LOGITS = sess.run([self.x_hat,self.logits],feed)
        #return X_hat, LOGITS
    
    def encode(self, X, layer= 'l4'):
        sess = tf.get_default_session()
        z = sess.run(self.layers[layer],{self.x:X})
        return z
    
    def load_weights(self, weights):
        ops = []
        if type(weights) == dict:
            for k, v in weights.iteritems():
                ops.append(
                    self.encoder_weights[k].assign(v)
                    )
        else:
            for i, w in enumerate(weights):
                if i % 2 == 0:
                    suffix = '_w'
                else:
                    suffix = '_b'
                ops.append(
                    self.encoder_weights[self.layer_names[i]+suffix].assign(w)
                )
        sess = tf.get_default_session()
        sess.run(ops)
    
class SplitNet(Model):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.supervised = True
        
        self.x = x = tf.placeholder(tf.float32, shape=(self.batch_size,84,84,4), name='x_input')
        self.xp = xp = tf.placeholder(tf.float32, shape=(self.batch_size,84,84,4), name='xp_input')
        
        self.y = y = tf.placeholder(tf.int32, shape=(self.batch_size,), name='y_input')
        
        with tf.variable_scope('encoding') as scope:
            z1 = self.encoder(x)
            scope.reuse_variables()
            z2 = self.encoder(xp)
            
        z = tf.concat(1,[z1,z2],name='z')
        self.logits = logits = self.classifier(z)
        self.x_hat = tf.argmax(self.logits,1)
        #self.x_hat = tf.to_int32(self.logits > 0)
        
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)
        )
        #self.cost = tf.reduce_mean(
            #tf.nn.sigmoid_cross_entropy_with_logits(tf.squeeze(logits),y)
        #)
        self.opt = tf.train.AdamOptimizer().minimize(self.cost)
        
        #self.l1_flat = tf.reshape(self.l1, [-1, reduce(lambda x, y: x * y, self.l1.get_shape().as_list()[1:])])
        
        
    def encoder(self,x):
        chans = [10,10,5]
        conv_w1 = tf.get_variable('conv_w1', shape=(3,3,4,chans[0]), dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer())
        conv_b1 = tf.get_variable('conv_b1', shape=(chans[0],), dtype=tf.float32)
        conv_w2 = tf.get_variable('conv_w2', shape=(5,5,chans[0],chans[1]), dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer())
        conv_b2 = tf.get_variable('conv_b2', shape=(chans[1],), dtype=tf.float32)
        conv_w3 = tf.get_variable('conv_w3', shape=(5,5,chans[1],chans[2]), dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer())
        conv_b3 = tf.get_variable('conv_b3', shape=(chans[2],), dtype=tf.float32)        
        
        h1 = activation_fn(tf.nn.conv2d(x, conv_w1, [1,1,1,1], "VALID") + conv_b1)
        h2 = activation_fn(tf.nn.conv2d(h1, conv_w2, [1,2,2,1], "VALID") + conv_b2)
        h3 = activation_fn(tf.nn.conv2d(h2, conv_w3, [1,2,2,1], "VALID") + conv_b3)
        h4 = h3
        #h4 = tf.nn.max_pool(h3, [1,2,2,1], [1,2,2,1], "VALID")
        h = tf.reshape(h4, [-1, reduce(lambda x, y: x * y, h4.get_shape().as_list()[1:])])
        
        self.merge_dim = h.get_shape()[-1].value
        
        self.encoder_weights = {'w1':conv_w1, 'w2':conv_w2, 'w3':conv_w3,
                                'b1':conv_b1, 'b2':conv_b2, 'b3':conv_b3}
        
        return h
        
    def classifier(self,z):
        w1 = tf.get_variable('w1',shape=(2 * self.merge_dim,512))
        b1 = tf.get_variable('b1',shape=(512,))
        w2 = tf.get_variable('w2',shape=(512,256))
        b2 = tf.get_variable('b2',shape=(256,))
        #w3 = tf.get_variable('w3',shape=(256,256))
        #b3 = tf.get_variable('b3',shape=(256,))
        
        w_out = tf.get_variable('w_out',shape=(256,2))
        b_out = tf.get_variable('b_out',shape=(2,))
        
        h1 = activation_fn(tf.nn.xw_plus_b(z, w1, b1, name='fc1'))
        h2 = activation_fn(tf.nn.xw_plus_b(h1, w2, b2, name='fc2'))
        #h3 = activation_fn(tf.nn.xw_plus_b(h2, w3, b3, name='fc2'))
        logits = tf.nn.xw_plus_b(h2, w_out, b_out, name='logits')
        
        self.decoder_weights = {'w1':w1,'w2':w2,'w_out':w_out,
                     'b1':b1,'b2':b2,'b_out':b_out}
        
        return logits


def train(model, X_t, Y_t, X_v, Y_v, epochs=1000, saver= None):
    sess = tf.get_default_session()
    N_t = X_t.shape[0]
    N_v = X_v.shape[0]
    
    for epoch in range(epochs):
        
        losses_t = []
        losses_v = []
        minibatches = range(0,N_t,model.batch_size)
        random.shuffle(minibatches)
        
        for i in minibatches:
            X_t_batch = X_t[i:model.batch_size+i]
            Y_t_batch = Y_t[i:model.batch_size+i]
            
            if model.supervised:
                X1_t_batch, X2_t_batch = np.split(X_t_batch,2,axis=1)
                
                #import matplotlib.pyplot as plt
                #i = 0
                #while True:
                    #f, (ax1) = plt.subplots(1)
                    #A = X1_t_batch[i,0,:,:,1].astype('float32')
                    #B = X2_t_batch[i,0,:,:,1].astype('float32')
                    #ax1.imshow(np.column_stack((A,B)))
                    #plt.show()
                    #print Y_t_batch[i]
                    
                    #i += 1                  
                    
                feed_t = {model.x: np.squeeze(X1_t_batch),
                          model.xp: np.squeeze(X2_t_batch),
                          model.y: Y_t_batch}
            else:
                feed_t = {model.x: X_t_batch, model.y : Y_t_batch}
                
            loss, _ = sess.run([model.cost,model.opt],feed_t)
            losses_t.append(loss)
            
        for i in range(0,N_v,model.batch_size):
            X_v_batch = X_v[i:model.batch_size+i]
            Y_v_batch = Y_v[i:model.batch_size+i]
             
            if model.supervised:
                X1_v_batch, X2_v_batch = np.split(X_v_batch,2,axis=1) 
                feed_v = {model.x: np.squeeze(X1_v_batch),
                          model.xp: np.squeeze(X1_v_batch),
                          model.y: Y_t_batch}
            else:
                feed_v = {model.x: X_v_batch, model.y : Y_t_batch}
                
            loss = sess.run(model.cost,feed_v)
            losses_v.append(loss) 
            
        loss_t = np.mean(losses_t)
        loss_v = np.mean(losses_v)
        
        print 'Epoch: {}/{}, Training loss: {}, validation loss: {}'.format(epoch,epochs,loss_t,loss_v)
        
        if saver is not None:
            #saver.save_models(epoch, [self])
            saver.save_dict(epoch, {'loss_t': loss_t, 'loss_v': loss_v})
            encoder_d = {key:tensor.eval(session=sess) for key, tensor in model.encoder_weights.iteritems()}
            decoder_d = {key:tensor.eval(session=sess) for key, tensor in model.decoder_weights.iteritems()}
            saver.save_dict(epoch, decoder_d, name= 'decoder')
            saver.save_dict(epoch, encoder_d, name= 'encoder')