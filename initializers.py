import numpy as np
import random
from shape_zoo import *

import itertools

def standard_initializer(n_blocks, shapes, sizes, rot_size, random_holes):

    block_selections= np.random.multinomial(n_blocks, [1./len(shapes)]*len(shapes))
    hDisp = [None]*len(shapes)
    bPers = [None]*len(shapes)
    hPers = [None]*len(shapes)
    bAngs = [None]*len(shapes)
    hAngs = [None]*len(shapes)

    assert len(block_selections) == len(shapes)

    canonical_positions = [[0.3,0.3],[0.3,0.7],[0.7,0.7],[0.7,0.3]]
    #canonical_positions = [[0.3,0.3],[0.3,0.7],[0.7,0.3]]
    #canonical_positions = [[0.3,0.5],[0.7,0.5]]
    if random_holes:
        random.shuffle(canonical_positions)

    for i, (shape_ix, n_b) in enumerate(zip(np.argsort(block_selections)[::-1], np.sort(block_selections)[::-1])):
        bPers[shape_ix] = np.around(np.random.uniform(0.05,0.95,(n_b,2)),1) # select block locations
        bAngs[shape_ix] = np.random.randint(0,360/rot_size,(n_b,)) * rot_size % 360 # select block angles

        try:
            hPers[shape_ix] = np.array([canonical_positions[i]])
            hAngs[shape_ix] = np.random.randint(0,360/rot_size,1) * rot_size % 360
            hDisp[shape_ix] = True
        except IndexError:
            hPers[shape_ix] = np.array([])
            hAngs[shape_ix] = np.array([])
            hDisp[shape_ix] = False

    D = [(shape, {'color':RED, # tuple
                  'hDisp':hDisp[i], # bool
                  'size':sizes[i], # int
                  'bPositions':bPers[i], # array, potentially empty
                  'hPositions':hPers[i], # array, potentially empty
                  'bAngles':bAngs[i], # array, potentially empty
                  'hAngles':hAngs[i] # array, potentially empty
                })
        for i, shape in enumerate(shapes)
        ]

    return D

def preference_initializer(n_blocks, shapes, sizes, rot_size, random_holes):
    """
    creates initial conditions for two blocks with no distractor holes,
    equidistant from appropriate holes. as described in the preference
    experiments.
    """
    block_selections = np.zeros(len(shapes)).astype('int32')
    z_ix= np.random.choice(range(len(shapes)),2,replace=False)
    block_selections[z_ix] = 1
    hDisp = [None]*len(shapes)
    bPers = [None]*len(shapes)
    hPers = [None]*len(shapes)
    bAngs = [None]*len(shapes)
    hAngs = [None]*len(shapes)

    assert len(block_selections) == len(shapes)

    canonical_positions = [[0.3,0.3],[0.3,0.7],[0.7,0.7],[0.7,0.3]]
    canonical_b_positions = [[0.3,0.5],[0.7,0.5]]
    random.shuffle(canonical_b_positions) # randomly shuffle block positions, else a side is favored.

    c = 0
    for i, (shape_ix, n_b) in enumerate(zip(np.argsort(block_selections)[::-1], np.sort(block_selections)[::-1])):
        if n_b > 0:
            bPers[shape_ix] = np.array([canonical_b_positions[c]]) # select block locations
            hPers[shape_ix] = np.array([canonical_positions[c],canonical_positions[c + 2]])
            hAngs[shape_ix] = np.random.randint(0,360/rot_size,2) * rot_size % 360
            hDisp[shape_ix] = True
            c += 1
        else:
            hPers[shape_ix] = np.array([])
            hAngs[shape_ix] = np.array([])
            hDisp[shape_ix] = False

            bPers[shape_ix] = np.array([])

        bAngs[shape_ix] = np.random.randint(0,360/rot_size,(n_b,)) * rot_size % 360 # select block angles

    D = [(shape, {'color':RED,
                  'hDisp':hDisp[i],
                  'size':sizes[i],
                  'bPositions':bPers[i],
                  'hPositions':hPers[i],
                  'bAngles':bAngs[i],
                  'hAngles':hAngs[i]
                })
        for i, shape in enumerate(shapes)
        ]

    return D

def grid_initializer(ix, shapes, sizes, rot_size, permute=0):
    """
    WARNING: The signature here is a little different. Creates a list of
    initial conditions, tiling the screen with a single block, tiling all
    orientations. Used to generate the datasets for supervised learning.
    """
    Ds = []
    if permute > 0:
        l = list(itertools.permutations(range(len(shapes)),4))
        random.shuffle(l)
        hole_orders = l[:permute]
    else:
        hole_orders = itertools.combinations(range(len(shapes)),4)

    for block_shape in shapes:
        for x_center in np.arange(0.1,1,0.1):
            for y_center in np.arange(0.1,1,0.1):
                if block_shape.angles_of_symmetry.sum() == 0.0:
                    block_angs = np.arange(0, 360, rot_size)
                else:
                    block_angs = np.arange(block_shape.angles_of_symmetry[0],
                                           block_shape.angles_of_symmetry[1],rot_size)
                block_angs = np.random.choice(block_angs,2,replace=False)
                for ang in block_angs:
                    #for hole_order in itertools.permutations(range(len(shapes)),4):
                    for hole_order in hole_orders:
                        hDisp = [None]*len(shapes)
                        bPers = [None]*len(shapes)
                        hPers = [None]*len(shapes)
                        bAngs = [None]*len(shapes)
                        hAngs = [None]*len(shapes)
                        canonical_positions = [[0.3,0.3],[0.3,0.7],[0.7,0.7],[0.7,0.3]]
                        random.shuffle(canonical_positions)
                        for shape_ix, can_pos in zip(hole_order, canonical_positions):
                            hPers[shape_ix] = np.array([can_pos])
                            hAngs[shape_ix] = np.random.randint(0,360/rot_size,1) * rot_size % 360
                            hDisp[shape_ix] = True
                        # create block
                        block_ix = shapes.index(block_shape)
                        #np.around(np.random.uniform(0.05,0.95,(n_b,2)),1)
                        for b_ix in range(len(shapes)):
                            if b_ix == block_ix:
                                bPers[b_ix] = np.array([[x_center,y_center]])
                                bAngs[b_ix] = np.array([ang])
                            else:
                                bPers[b_ix] = np.array([])
                                bAngs[b_ix] = np.array([])
                        D = [(shape, {'color':RED, # tuple
                                      'hDisp':hDisp[i], # bool
                                      'size':sizes[i], # int
                                      'bPositions':bPers[i], # array, potentially empty
                                      'hPositions':hPers[i], # array, potentially empty
                                      'bAngles':bAngs[i], # array, potentially empty
                                      'hAngles':hAngs[i] # array, potentially empty
                                    })
                            for i, shape in enumerate(shapes)
                            ]

                        Ds.append(D)

    return Ds
