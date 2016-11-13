from game_settings import *

from shapesorting import *
from sandbox.util import Saver

import argparse
import numpy as np
import sklearn as sk
import sklearn.metrics as mets

import sklearn.linear_model
import sklearn.lda
import sklearn.naive_bayes
import sklearn.tree

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
#parser.add_argument('--encoding_time',type=str,default='16-11-11-07-17PM')
parser.add_argument('--encoding_time',type=str,default='16-11-11-07-18PM')
#parser.add_argument('--encoding_time',type=str,default='16-11-11-08-31PM')

parser.add_argument('--classification',type=bool,default=False)
parser.add_argument('--similarity',type=bool,default=False)
args = parser.parse_args()

encoding_saver = Saver(time=args.encoding_time,path='{}/{}'.format(DATADIR,'enco_simi_data'))

simi_data = encoding_saver.load_dictionary(0,'simi_data')
Y = simi_data['Y']
X1 = simi_data['X1']
SHAPES1 = simi_data['SHAPES1']

#Y = encoding_saver.load_value(0,'Y')
D = encoding_saver.load_dictionary(0, 'l3_flat_encodings') # more overfitting, more validation accuracy
D = encoding_saver.load_dictionary(0, 'l2_flat_encodings') 
#D = encoding_saver.load_dictionary(0, 'adv_hid_encodings')

if args.classification:
    models = [sk.linear_model.LogisticRegression(), sk.svm.SVC(), sklearn.lda.LDA(),
              sklearn.naive_bayes.GaussianNB(), sklearn.tree.DecisionTreeClassifier()]
    
    X = D['rZ1']
    N = X.shape[0]
    cutoff = int(0.7 * N)

    O_t = X1[:cutoff]
    O_v = X1[cutoff:]

    X_t = X[:cutoff]
    Y_t = SHAPES1[:cutoff]
    X_v = X[cutoff:]
    Y_v = SHAPES1[cutoff:]    
    
    for model in models:
        
        #for i in range(O_t.shape[0]):
            #plt.imshow(O_t[i])
            #print SHAPESORT_ARGS1['shapes'][Y_t[i]]
            #plt.show()
        
        model.fit(X_t,Y_t)
        
        print model
        print "train: {}".format(mets.accuracy_score(Y_t, model.predict(X_t)))
        print "valid: {}".format(mets.accuracy_score(Y_v, model.predict(X_v)))    

if args.similarity:
    models = [sk.linear_model.LogisticRegression(), sk.svm.SVC(), sklearn.lda.LDA(),
              sklearn.naive_bayes.GaussianNB(), sklearn.tree.DecisionTreeClassifier()]
    
    for merge in [lambda x, y : np.column_stack((x,y)), lambda x, y : x * y]:
        
        X = merge(D['rZ1'],D['rZ2'])
        N = X.shape[0]
        p = np.random.permutation(N)
        
        X = X[p]
        Y = Y[p]
        cutoff = int(0.7 * N)
        
        X_t = X[:cutoff]
        Y_t = Y[:cutoff]
        X_v = X[cutoff:]
        Y_v = Y[cutoff:]    
        
        for model in models:
            print str(type(model))
            model.fit(X_t, Y_t)
            
            print mets.accuracy_score(Y_t, model.predict(X_t))
            print mets.accuracy_score(Y_v, model.predict(X_v))

halt= True