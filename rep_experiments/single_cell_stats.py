from game_settings import *

from shapesorting import *
from sandbox.util import Saver

import argparse
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt

import itertools

parser = argparse.ArgumentParser()
#parser.add_argument('--encoding_time',type=str,default='16-11-11-07-17PM')
parser.add_argument('--encoding_time',type=str,default='16-11-11-07-18PM')
#parser.add_argument('--encoding_time',type=str,default='16-11-11-08-31PM')

parser.add_argument('--classification',type=bool,default=False)
parser.add_argument('--similarity',type=bool,default=False)
parser.add_argument('--encoding',type=str,default="adv_hid_encodings")

parser.add_argument('--categories',type=str,default='shape')

args = parser.parse_args()

encoding_saver = Saver(time=args.encoding_time,path='{}/{}'.format(DATADIR,'enco_simi_data'))

simi_data = encoding_saver.load_dictionary(0,'simi_data')
Y = simi_data['Y']
X1 = simi_data['X1']
SHAPES1 = simi_data['SHAPES1']
ANGLES1 = simi_data['ANGLES1']

#Y = encoding_saver.load_value(0,'Y')
#D = encoding_saver.load_dictionary(0, 'l3_flat_encodings') # more overfitting, more validation accuracy
#D = encoding_saver.load_dictionary(0, 'l2_flat_encodings') 
#D = encoding_saver.load_dictionary(0, 'adv_hid_encodings')
D = encoding_saver.load_dictionary(0, args.encoding)

X = D['rZ1']
Xshapes = []
for s in np.unique(SHAPES1):
    Xshapes.append(
        X[SHAPES1 == s]
    )
    
sts = []

class LayerProfile(object):
    def __init__(self):
        self._neuron_profiles = []
        
    @property
    def neuron_profiles(self):
        return self._neuron_profiles
    
    @neuron_profiles.setter
    def neuron_profiles(self,value):
        self._neuron_profiles = value

    @property
    def average_responsiveness(self):
        return np.mean([npp.num_responsive for npp in self.neuron_profiles])    

class NeuronProfile(object):
    def __init__(self):
        self.responsive_pairs = set()
        
    @property
    def num_responsive(self):
        return len(self.responsive_pairs)

layer_profile = LayerProfile()
neuron_profiles = []
for neuron in xrange(X.shape[1]):
    print "{} of {}".format(neuron,X.shape[1])
    neuron_profile = NeuronProfile()
    
    sts.append(
        stats.f_oneway(*[x[:,neuron] for x in Xshapes])
        )
    pvals = [st.pvalue for st in sts]
    significant = [pval < 0.05 for pval in pvals]
    
    pairs = [pair for pair in itertools.combinations(np.unique(SHAPES1), 2) if pair[0] != pair[1]]
    #for sig in significant:
        #if sig:
    ttest_pvals = []
    ttest_significant = []
    for pair in pairs:
        ttest_result = stats.ttest_ind(Xshapes[pair[0]][:,neuron],Xshapes[pair[1]][:,neuron])
        ttest_pval = ttest_result.pvalue
        
        ttest_pvals.append(ttest_pval)
        if ttest_pval < 0.05:
            neuron_profile.pval = ttest_pval
            neuron_profile.responsive_pairs.add(pair)
                        
    neuron_profiles.append(neuron_profile)
    
layer_profile.neuron_profiles = neuron_profiles
         
halt= True