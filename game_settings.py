from shape_zoo import *

DISCRETE_ACT_MAP1 = \
    [['none'],
     ['up'],
     ['up', 'right'],
     ['right'],
     ['right', 'down'],
     ['down'],
     ['down', 'left'],
     ['left'],
     ['up', 'left'],
     ['grab'],
     ['grab','up'],
     ['grab','up', 'right'],
     ['grab','right'],
     ['grab','right', 'down'],
     ['grab','down'],
     ['grab','down', 'left'],
     ['grab','left'],
     ['grab','up', 'left']     
     ]

DISCRETE_ACT_MAP2 = \
    [['none'],
     ['up'],
     ['up', 'right'],
     ['right'],
     ['right', 'down'],
     ['down'],
     ['down', 'left'],
     ['left'],
     ['up', 'left'],
     ['grab'],    
     ]

DISCRETE_ACT_MAP3 = \
    [['up'],
     ['right'],
     ['down'],
     ['left'],
     ['grab'],    
     ]

DISCRETE_ACT_MAP4 = \
    [['up'],
     ['right'],
     ['down'],
     ['left'],
     ['grab'],    
     ['rotate_cw'],
     ['rotate_ccw']
     ]

REWARD_DICT1 = \
    {'boundary':-0.1,
     'hold_block':0.1,
     'fit_block':1000.0,
     'trial_end':5000.0
    }

REWARD_DICT2 = \
    {'boundary':-0.001,
     'hold_block':0.001,
     'fit_block':10.0,
     'trial_end':50.0
    }

REWARD_DICT3 = \
    {'boundary':-1,
     'hold_block':0.001,
     'fit_block':10.0,
     'trial_end':50.0
    }

SHAPESORT_ARGS0 = dict(
        act_mode='discrete',
        grab_mode='toggle',
        shapes=[Trapezoid, RightTri, Hexagon, Tri, Rect],
        sizes=[60,60,60,60,60],
        n_blocks=3,
        random_cursor=True,
        random_holes=True,
        step_size=20,
        rot_size=360
    )

SHAPESORT_ARGS1 = dict(
        act_mode='discrete',
        grab_mode='toggle',
        shapes=[Trapezoid, RightTri, Hexagon, Tri, Rect],
        sizes=[60,60,60,60,60],
        n_blocks=3,
        random_cursor=True,
        random_holes=True,
        step_size=20,
        rot_size=30,
        screen_HW=200,
        screen_rHW=84,
        cursor_size=10
    )

#SHAPESORT_ARGS2 = dict(
        #act_mode='discrete',
        #grab_mode='toggle',
        #shapes=[Trapezoid, RightTri, Hexagon, Tri, Rect],
        #sizes=[20,20,20,20,20],
        #n_blocks=3,
        #random_cursor=True,
        #random_holes=True,
        #step_size=8,
        #rot_size=30,
        #screen_HW=80,
        #screen_rHW=42,
        #cursor_size=5
    #)

SHAPESORT_ARGS2 = []

#SHAPESORT_ARGS2 = dict(
        #act_mode='discrete',
        #grab_mode='toggle',
        #shapes=[Trapezoid, RightTri, Hexagon, Tri, Rect] * 2,
        #sizes=[60,60,60,60,60] + [40,40,40,40,40],
        #n_blocks=3,
        #random_cursor=True,
        #random_holes=True,
        #step_size=20,
        #rot_size=30
    #)

#SHAPESORT_ARGS3 = dict(
        #act_mode='discrete',
        #grab_mode='toggle',
        #shapes=[Hexagon, Tri, Rect],
        #sizes=[60,60,60],
        #n_blocks=3,
        #random_cursor=True,
        #random_holes=True,
        #step_size=20,
        #rot_size=30,
        #act_map= DISCRETE_ACT_MAP4,
        #reward_dict= REWARD_DICT3
    #)

#SHAPESORT_ARGS4 = dict(
        #act_mode='discrete',
        #grab_mode='toggle',
        #shapes=[Hexagon, Tri, Rect],
        #sizes=[60,60,60],
        #n_blocks=4,
        #random_cursor=False,
        #random_holes=True,
        #step_size=25,
        #rot_size=45,
        #shrink_hw=32,
        #act_map= DISCRETE_ACT_MAP4,
        #reward_dict= REWARD_DICT2
    #)

SHAPESORT_ARGS = [SHAPESORT_ARGS0, SHAPESORT_ARGS1, SHAPESORT_ARGS2]
