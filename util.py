import gym
from game_settings import SHAPESORT_ARGS0, SHAPESORT_ARGS1, SHAPESORT_ARGS2
from game import ShapeSorter

import numpy as np

import os
import h5py
import json

import datetime
import calendar

class ShapeSorterWrapper(ShapeSorter):

    def __init__(self):
        super(ShapeSorterWrapper, self).__init__(
            act_mode=ShapeSorterWrapper.act_mode,
            grab_mode=ShapeSorterWrapper.grab_mode,
            shapes=ShapeSorterWrapper.shapes,
            sizes=ShapeSorterWrapper.sizes,
            n_blocks=ShapeSorterWrapper.n_blocks,
            random_cursor=ShapeSorterWrapper.random_cursor,
            random_holes=ShapeSorterWrapper.random_holes,
            step_size=ShapeSorterWrapper.step_size,
            rot_size=ShapeSorterWrapper.rot_size,
            act_map=ShapeSorterWrapper.act_map,
            reward_dict=ShapeSorterWrapper.reward_dict
        )

    @classmethod
    def set_initials(cls, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(ShapeSorterWrapper,k,v)

ShapeSorterWrapper.set_initials(**SHAPESORT_ARGS1)

def register_env():
    gym.envs.register(
        id= "ShapeSorter-v0",
        #entry_point='rltools.envs.julia_sim:FollowingWrapper',
        entry_point='register_shapesorter:ShapeSorterWrapper',
        timestep_limit=15000,
        reward_threshold=15000,
    )

    env = gym.make('ShapeSorter-v0')
    return env

class Saver(object):
    def __init__(self,path=None,overwrite= False,time=None):
        if time is None:
            now = datetime.datetime.now()
            self.time = calendar.datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S-%f')
        else:
            self.time = time

        if path is None:
            path = config.LOG_DIR
        assert path[-1] != "/"
        self._path = path + "/" + self.time + "/"

    def save_args(self,args):
        #with h5py.File(self._path + "args.txt", 'a') as hf:
        self.dircheck()
        json.dump(vars(args), open(self._path + "args.txt",'a'))

    def load_args(self):
        with open(self._path + "args.txt","r") as f:
            d = eval(f.readlines()[0])
        return d

    def save_dict(self,itr, d, name= ""):
        self.dircheck()
        with h5py.File(self._path + "epochs.h5", 'a') as hf:
            for key, value in d.iteritems():
                hf.create_dataset("iter{itr:05}{name}/".format(itr=itr,name="/"+name)+key,data=value)

    def save_recursive_dict(self, itr, d, name= ""):
        def recurse(f, p, x):
            for key, item in x.items():
                if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, list)):
                    f[p + key] = item
                elif isinstance(x, dict):
                    recurse(hf, p + key + '/', item)
                else:
                    raise ValueError('Cannot save %s type'%type(item))

        path = self._path
        with h5py.File(path + "epochs.h5", 'a') as hf:
            recurse(hf, name+"/", d)

            halt= True

    def save_models(self,itr,models):
        self.dircheck()
        with h5py.File(self._path + "epochs.h5",'a') as hf:
            for model in models:
                tensors = model.weights
                weights = model.get_weights()
                for tensor, weight in zip(tensors, weights):
                    hf.create_dataset("iter{itr:05}/{model}/{tensor}".format(itr=itr,model=model.name,tensor=tensor.name),data = weight)

    def load_models(self,itr,models):
        with h5py.File(self._path + "epochs.h5",'r') as hf:
            keys = hf.keys()
            for model in models:
                saved_model= hf[keys[itr]][model.name]
                L = []
                for w in model.weights:
                    saved_w = saved_model[w.name]
                    L.append(saved_w)

                model.set_weights(L)

        return models

    def load_value(self,itr,key):
        with h5py.File(self._path + "epochs.h5",'r') as hf:
            value = hf['iter{itr:05}'.format(itr=itr)][key][...]
        return value

    def load_dictionary(self,itr,name):
        with h5py.File(self._path + "epochs.h5",'r') as hf:
            D = {k:v[...] for k, v in hf['iter{itr:05}'.format(itr=itr)][name].iteritems()}
        return D

    def load_recursive_dictionary(self, name):
        """
        ....
        """
        def recurse(f, p):
            ans = {}
            for key, item in f[p].items():
                if isinstance(item, h5py._hl.dataset.Dataset):
                    ans[key] = item.value
                elif isinstance(item, h5py._hl.group.Group):
                    ans[key] = recurse(f, p + key + '/')
            return ans

        with h5py.File(self._path + "epochs.h5",'r') as hf:
            D = recurse(hf, name + "/")

        return D

    def dircheck(self):
        if not os.path.isdir(self._path):
            os.mkdir(self._path)

def vis(path):
    with h5py.File(path,'r') as hf:
        X = h5

