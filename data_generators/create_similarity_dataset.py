from shapesorting import *

import argparse
from shape_zoo import *
from game_settings import SHAPESORT_ARGS, INITIALIZER_MAP

import matplotlib.pyplot as plt

from game import process_observation, create_renderList
import pygame as pg

from sandbox.util import Saver

parser = argparse.ArgumentParser()
parser.add_argument('--vary_pos',type=bool,default=False)
parser.add_argument('--vary_ang',type=bool,default=True)
parser.add_argument('--vary_size',type=bool,default=False)

parser.add_argument('--shapesort_args',type=int,default=1)
parser.add_argument('--shapes',type=str,nargs='+')
parser.add_argument('--sizes',type=int,nargs='+')

parser.add_argument('--size_lb',type=int,default=35)
parser.add_argument('--size_ub',type=int,default=85)

parser.add_argument('--grab',type=int,default=0)
parser.add_argument('--hole',type=int,default=0)

parser.add_argument('--N',type=int,default=50000) # size of dataset

args = parser.parse_args()

if args.shapesort_args is None:
    assert args.shape is not None and args.sizes is not None
    assert len(args.shapes) == len(args.sizes)
else:
    shapes = SHAPESORT_ARGS[args.shapesort_args]['shapes']
    sizes = SHAPESORT_ARGS[args.shapesort_args]['sizes']
    rot_size = SHAPESORT_ARGS[args.shapesort_args]['rot_size']
    step_size = SHAPESORT_ARGS[args.shapesort_args]['step_size']
    cursor_size = SHAPESORT_ARGS[args.shapesort_args]['cursor_size']
    HW = SHAPESORT_ARGS[args.shapesort_args]['screen_HW']    
    rHW = SHAPESORT_ARGS[args.shapesort_args]['screen_rHW']
    
X1 = []
X2 = []
Y = []

SHAPES1 = []
SIZES1 = []
ANGLES1 = []
POSITS1 = []

SHAPES2 = []
SIZES2 = []
ANGLES2 = []
POSITS2 = []

H = 200
W = 200

if args.hole == 0:
    shape_col = RED
elif args.hole == 1:
    shape_col = BLACK
else:
    raise NotImplementedError

if args.grab == 0:
    col = RED
elif args.grab == 1:
    col = GREEN
elif args.grab == 2:
    col = BLUE
else:
    raise NotImplementedError

for i in range(args.N):
    print "{} of {}".format(i,args.N)
    screen = pg.display.set_mode((H, W))
    screen.fill(WHITE)  
    
    ix_1 = np.random.choice(range(len(shapes)))
    if i < args.N/2: # generate similar datapoints
        ix_2 = ix_1
        Y.append(1)
    else:
        ix_2 = np.random.choice(range(len(shapes)))
        while ix_2 == ix_1:
            ix_2 = np.random.choice(range(len(shapes)))
        Y.append(0)
            
    shape1 = shapes[ix_1]
    shape2 = shapes[ix_2]
    size1 = sizes[ix_1]
    size2 = sizes[ix_2]
    
    if args.vary_ang:
        ang1 = np.random.randint(0,360)
        ang2 = np.random.randint(0,360)
    else:
        ang1 = 0
        ang2 = 0
        
    if args.vary_pos:
        t = 60
        offset1 = np.array([np.random.randint(-t,t),np.random.randint(-t,t)])
        offset2 = np.array([np.random.randint(-t,t),np.random.randint(-t,t)])
    else:
        offset1 = 0.0
        offset2 = 0.0
        
    if args.vary_size:
        size1 = np.random.randint(args.size_lb,args.size_ub) # 40, 70
        size2 = np.random.randint(args.size_lb,args.size_ub)
    
    # generate first image
    center1 = np.array([H,W])/2. + offset1
    block1 = shape1(shape_col, center1, size1, 'block', angle=ang1)
    block1.render(screen)
    pg.draw.circle(screen, col, center1.astype('int32'), cursor_size)
    
    x1 = process_observation(screen,HW,rHW)[None,...]
    X1.append(x1)
    
    screen.fill(WHITE)
    
    # generate second image
    center2 = np.array([H,W])/2. + offset2
    block2 = shape2(shape_col, center2, size2, 'block', angle=ang2)
    block2.render(screen)
    pg.draw.circle(screen, col, center2.astype('int32'), cursor_size)    
    
    x2 = process_observation(screen,HW,rHW)[None,...]
    X2.append(x2)
        
    SHAPES1.append(shapes.index(shape1))
    SIZES1.append(size1)
    ANGLES1.append(ang1)
    POSITS1.append(offset1)
    
    SHAPES2.append(shapes.index(shape2))
    SIZES2.append(size2)
    ANGLES2.append(ang2)
    POSITS2.append(offset2)    
    
    if False:
        xband = np.column_stack((x1, np.zeros_like(x1), x2))
        plt.imshow(xband)
        plt.show()
        
X1 = np.concatenate(X1,axis=0)
X2 = np.concatenate(X2,axis=0)
Y = np.array(Y)

saver = Saver(path="{}/{}".format(DATADIR,'simi_data'))

saver.save_args(args)
saver.save_dict(0,{'X1':X1,
                   'X2':X2,
                   'Y':Y,
                   'SHAPES1':SHAPES1,
                   'SIZES1':SIZES1,
                   'ANGLES1':ANGLES1,
                   'POSITS1':POSITS1,
                   'SHAPES2':SHAPES2,
                   'SIZES2':SIZES2,
                   'ANGLES2':ANGLES2,
                   'POSITS2':POSITS2},name='data')

args = parser.parse_args()