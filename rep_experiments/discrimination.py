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

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_time',type=str,default='16-11-06')
args = parser.parse_args()

encoding_saver = Saver(time=args.encoding_time,path='{}/{}'.format(DATADIR,'enco_simi_data'))

Y = encoding_saver.load_value(0,'Y')
D = encoding_saver.load_dictionary(0, 'adv_hid_encodings')

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