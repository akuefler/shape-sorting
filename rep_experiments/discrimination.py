import argparse
import numpy as np

import sklearn
from sklearn.decomposition import PCA
import sklearn.svm
import sklearn.lda
import sklearn.linear_model

import sklearn.metrics as mets

import matplotlib.pyplot as plt

from config import DATADIR

from util import Saver

import tqdm

parser = argparse.ArgumentParser()

## includes holes
#parser.add_argument("--data_time",type=str,default="17-01-19-07-59PM") # no grab
#parser.add_argument("--data_time",type=str,default="17-01-19-09-27PM") # grab 1
#parser.add_argument("--data_time",type=str,default="17-01-19-09-29PM") # grab 2

#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-19-07-59PM",
                                                                  #"17-01-19-09-27PM",
                                                                  #"17-01-19-09-29PM"])

## no holes
#parser.add_argument("--data_time",type=str,default="17-01-19-09-35PM") # no grab
#parser.add_argument("--data_time",type=str,default="17-01-19-09-36PM") # grab 1
#parser.add_argument("--data_time",type=str,default="17-01-19-09-37PM") # grab 2

#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-19-09-35PM",
                                    #"17-01-19-09-36PM",
                                    #"17-01-19-09-37PM"])

# Fixed Position
#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-19-22-34-07-174659",
                                    #"17-01-19-22-34-56-056347",
                                    #"17-01-19-22-35-45-287050"])

## Fixed Position, No Holes
#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-19-22-36-33-368248",
                                    #"17-01-19-22-37-16-244572",
                                    #"17-01-19-22-37-58-907670"])
## Holes, Enumerated
#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-20-21-12-51-477494",
                                    #"17-01-20-21-13-10-048480",
                                    #"17-01-20-21-13-35-095721"])

## No Holes, Enumerated
#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-20-21-55-33-448895",
                                    #"17-01-20-21-56-02-187089",
                                    #"17-01-20-21-56-21-432451"])
### No Holes, Enumerated
#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-20-22-42-10-413497",
                                    #"17-01-20-22-42-48-884555",
                                    #"17-01-20-22-43-27-180222"])

"""
Data used in original manuscript.
"""
### 81,000, holes, enumerated
#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-20-23-17-53-444875",
#                                    "17-01-20-23-26-23-488455"]),

                                    ##"17-01-20-23-33-47-043926"])

### 81,000, holes, enumerated # RANDOM NETWORK
parser.add_argument("--data_times", nargs="+", type=str, default=[
                                    "17-01-29-14-21-59-644400",
                                    "17-01-29-14-30-54-397244"])

                                    #"17-01-20-23-33-47-043926"])

parser.add_argument("--encodings",nargs="+",type=str,default=["Z_l1_flat","Z_l2_flat","Z_l3_flat","Z_value_hid","Z_adv_hid"])

parser.add_argument("--cutoff",type=float,default=0.25)
parser.add_argument("--n_features",type=int,default=300)
parser.add_argument("--feat_ix",type=int,default=0)
parser.add_argument("--dim_reduce",type=int,default=1)
parser.add_argument("--normalize",type=int,default=0)

parser.add_argument("--models",type=str,nargs="+",default=["softmax","svm","lda"])

args = parser.parse_args()

disc_saver = Saver(path='{}/{}'.format(DATADIR,'disc_results'))
data_savers = [Saver(time=data_time,path='{}/{}'.format(DATADIR,'scene_and_enco_data'))
               for data_time in args.data_times]

_models = {'softmax':sklearn.linear_model.LogisticRegression(multi_class='multinomial',solver='sag'),
          'svm':sklearn.svm.LinearSVC(dual=False, C=0.1),
          'lda':sklearn.lda.LDA()}
models = [(name, classif) for name, classif in _models.iteritems() if name in args.models]

# weight matrix for each classifer
n_class = 5
W = np.zeros((len(data_savers), len(args.encodings), len(models)) + (n_class, args.n_features,))
B = np.zeros((len(data_savers), len(args.encodings), len(models)) + (n_class,))

# classifier accuracy
MT = np.zeros((len(data_savers), len(args.encodings), len(models)))
MV = np.zeros((len(data_savers), len(args.encodings), len(models)))

# classifier confusion
CT = np.zeros((len(data_savers), len(args.encodings), len(models), n_class, n_class))
CV = np.zeros((len(data_savers), len(args.encodings), len(models), n_class, n_class))

c = 0
C = len(data_savers) * len(args.encodings) * len(models)
for i, data_saver in enumerate(data_savers):
    exp_args = data_saver.load_args()

    data = data_saver.load_dictionary(0,"data")

    # extract dataset information
    if i == 0:
        X = data["X"]
        N = X.shape[0]
        p = np.random.permutation(N)
        cutoff = int(args.cutoff * len(p))

        p_t = p[:cutoff]
        p_v = p[cutoff:]

    blocks = data_saver.load_recursive_dictionary("blocks")
    BLOCK_SHAPE = np.concatenate([np.array(blocks['{:0{}}'.format(itr,len(str(N)))]['shape']) for itr in range(N)])

    for j, layer in enumerate(args.encodings):
        # create training and validation sets
        Z_t = data[layer][p_t]
        y_t = BLOCK_SHAPE[p_t]

        Z_v = data[layer][p_v]
        y_v = BLOCK_SHAPE[p_v]

        if args.normalize:
            Z_t_mean = Z_t.mean(axis=0)
            Z_t_stdv = Z_t.std(axis=0) + 1e-6
            Z_t -= Z_t_mean
            Z_t /= Z_t_stdv
            Z_v -= Z_t_mean
            Z_v /= Z_t_stdv

        pca = PCA()
        if args.dim_reduce:
            G_t = pca.fit_transform(Z_t)[:,args.feat_ix:args.feat_ix+args.n_features]
            G_v = pca.transform(Z_v)[:,args.feat_ix:args.feat_ix+args.n_features]
        else:
            G_t = Z_t
            G_v = Z_v

        assert G_t.shape[-1] == args.n_features

        for k, (model_name, model) in enumerate(models):
            #print(model_name)
            model.fit(G_t,y_t)
            #print("Encoding: {}".format(layer))
            acc_t = mets.accuracy_score(y_t, model.predict(G_t))
            cm_t = mets.confusion_matrix(y_t, model.predict(G_t), labels=range(n_class))
            #print "train: {}".format(acc_t)
            acc_v = mets.accuracy_score(y_v, model.predict(G_v))
            cm_v = mets.confusion_matrix(y_v, model.predict(G_v), labels=range(n_class))

            MT[i,j,k], MV[i,j,k] = acc_t, acc_v
            CT[i,j,k], CV[i,j,k] = cm_t, cm_v
            W[i,j,k] = model.coef_
            B[i,j,k] = model.intercept_

            c+= 1
            if c % 5 == 0:
                print("{} of {}...".format(c,C))

disc_saver.save_dict(0, {"MT":MT, "CT":CT, "MV":MV, "CV":CV,
                         "W":W, "B":B, "N":N, "p":p}, name="data")
disc_saver.save_args(args)
