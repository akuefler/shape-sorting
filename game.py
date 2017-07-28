import pygame as pg
from pygame import Surface
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from scipy.stats import entropy
import matplotlib as mpl

import h5py

from config import *

import sys
import gym
from gym.spaces import Discrete, Box

from game_settings import DISCRETE_ACT_MAP4 as DISCRETE_ACT_MAP
from game_settings import REWARD_DICT2 as REWARD_DICT
from game_settings import INITIALIZER_MAP

from math import pi

import shapely.geometry
from shape_zoo import *

import time
import random

def angular_distance(x,matches,theta):
    return np.min([np.abs((x - h_ang + 180) % 360 - 180) for h_ang in matches if h_ang % theta == 0])

def fit(hole, block, TOL= 10):
    """
    determine if block fits in hole.

    Args:
        hole: hole object
        block: block object

    Returns:
        does_fit: boolean indicating if the block and hole fit together

    """
    pos_fit= np.linalg.norm(np.array(hole.center)-
               np.array(block.center)) < TOL
    U = shapely.geometry.asPolygon(hole.vertices)
    V = shapely.geometry.asPolygon(block.vertices)

    geom_fit = U.contains(V)
    does_fit = pos_fit and geom_fit
    return does_fit

def process_observation(screen,HW,rHW):
    """
    Resizes an image of size HW down to size rHW, additionally "standardizes"
    the image.
    """
    r = pg.surfarray.pixels_red(screen).astype('float32')
    g = pg.surfarray.pixels_green(screen).astype('float32')
    b = pg.surfarray.pixels_blue(screen).astype('float32')
    X = (2 ** 2) * r + 2 * g + b # convert to 1D map
    if HW != rHW:
        Z = imresize(X,(rHW,rHW))
    else:
        Z = X
    Y = (Z.astype('float32') - 255/2) / (255/2)

    return Y

def create_renderList(specs, H, W):
    """
    Generates lists of hole and block objects to be rendered according to specs

    Args:
        specs: a list of tuples (cls, d) where "cls" is a shape class (e.g.,
        shape_zoo.Trapezoid) and "d" is a dictionary containing arguments for
        instantiating objects of that cls on screen (e.g., "hPositions" gives a set
        of coordinates specifying the number, and posiiton, of holes of that type
        to produce).

        H, W: height and width of the pygame screen

    Returns:
        holeList: List of hole objects.
        blockList: List of block objects.
    """
    blockList = []
    holeList = []
    for key, val in specs:

        obj = key
        kwargs = {key : val for key, val in val.iteritems()
                  if key not in ['bPositions', 'hPositions', 'bAngles', 'hAngles']}

        for i, per in enumerate(val['bPositions']):
            kwargs['center'] = (int(per[0]*H), int(per[1]*W))
            kwargs['typ'] = 'block'
            if 'bAngles' in val.keys():
                kwargs['angle']= val['bAngles'][i]
            blockList.append(obj(**kwargs))

        if val['hDisp']:
            del val['hDisp']
            for i, per in enumerate(val['hPositions']):
                kwargs['center'] = (int(per[0]*H), int(per[1]*W))
                kwargs['typ'] = 'hole'
                kwargs['color'] = BLACK
                if 'hAngles' in val.keys():
                    kwargs['angle']= val['hAngles'][i]
                kwargs['size'] = val['size'] + 5
                holeList.append(obj(**kwargs))

    return holeList, blockList

class ShapeSorter(object):
    """
    """
    def __init__(self, grab_mode= 'toggle',
                 shapes = [Trapezoid, RightTri, Hexagon, Tri, Rect, Star],
                 sizes = [60,60,60,60,60,60],
                 random_cursor= False,
                 random_holes= True,
                 n_blocks = 3,
                 act_map = DISCRETE_ACT_MAP,
                 reward_dict = REWARD_DICT,
                 step_size = 20,
                 rot_size = 30,
                 screen_HW = 200,
                 screen_rHW = 84,
                 cursor_size = 10
                 ):
        assert len(sizes) == len(shapes)
        pg.init()

        self.H, self.W = screen_HW, screen_HW
        self.rHW = screen_rHW
        self.cursor_size = cursor_size

        self.act_map = act_map
        self.reward_dict = reward_dict

        self.step_size = step_size
        self.rot_size = rot_size

        self.shapes = shapes
        self.sizes = sizes
        self.n_blocks = n_blocks

        self.screen=pg.display.set_mode((self.H, self.W))
        self.screenCenter = (self.H/2,self.W/2)
        self.grab_mode= grab_mode

        self.random_holes = random_holes
        self.random_cursor = random_cursor

        # create gym spaces for actions and observations
        self.action_space = Discrete(len(self.act_map))
        self.observation_space = Box(-float('inf'),float('inf'),(self.rHW,self.rHW))

        # create initializer, determining where blocks/holes are
        # initialized in the MDP
        self.initialize()

    def initialize(self):
        """
        initializes the state of the MDP at the beginning of the episode: e.g.,
        the holes and blocks present, the current x and y velocity of the
        cursor, etc.

        Args:
            initial_conditions: a list of tuples (cls, d) where "cls" is a shape class (e.g.,
            shape_zoo.Trapezoid) and "d" is a dictionary containing arguments for
            instantiating objects of that cls on screen (e.g., "hPositions" gives a set
            of coordinates specifying the number, and posiiton, of holes of that type
            to produce).

            see create_renderList

        """
        self.state= {}
        self.state['x_speed'] = 0
        self.state['y_speed'] = 0
        if self.random_cursor:
            self.state['cursorPos'] = np.array([np.random.randint(self.W*0.1, self.W - 0.1*self.W),
                                                                  np.random.randint(self.H*0.1, self.H - 0.1*self.H)])
        else:
            self.state['cursorPos'] = self.screenCenter

        # randomly generates a configuration of blocks and holes according to
        # the chosen initializer
        initial_conditions = INITIALIZER_MAP["training"](self.n_blocks, self.shapes,
            self.sizes, self.rot_size, self.random_holes)
        hList, bList = create_renderList(initial_conditions, self.H, self.W)

        self.state['hList'] = hList
        self.state['bList'] = bList
        self.state['grab'] = False
        self.state['target'] = None

    def step(self, action):
        """
        Updated the current state of the shape sorting MDP by applying an
        action.

        Args:
            action: An integer index into the action_map, representing a
            cursor movement, rotation, or grab

        Returns:
            observation: An image representing the game screen.
            reward: a scalar reward signal
            info: dictionary containing additional information about the
            timestep.

        see https://gym.openai.com/ for details.

        """
        info = {}
        reward = 0.0
        done = False
        prevCursorPos = self.state['cursorPos']
        penalize = False
        self.screen.fill(WHITE)
        act_category = -1
        if type(action) != list:
            agent_events = self.act_map[action]
        else:
            agent_events = action

        if self.grab_mode != 'toggle':
            if 'grab' in agent_events:
                self.state['grab'] = True
                act_category = 0
            else:
                self.state['grab'] = False
        else:
            if 'grab' in agent_events:
                self.state['grab'] = not self.state['grab']
                act_category = 0

        if 'left' in agent_events:
            self.state['x_speed'] = x_speed = -self.step_size
            act_category = 1

        elif 'right' in agent_events:
            self.state['x_speed'] = x_speed = self.step_size
            act_category = 1

        else:
            self.state['x_speed'] = x_speed = 0
        if 'up' in agent_events:
            self.state['y_speed'] = y_speed = -self.step_size
            act_category = 1

        elif 'down' in agent_events:
            self.state['y_speed'] = y_speed = self.step_size
            act_category = 1

        else:
            self.state['y_speed'] = y_speed = 0

        (x_pos, y_pos) = self.state['cursorPos']
        self.state['cursorPos'] = cursorPos = (int(max([min([x_pos + x_speed, self.H - 0.1*self.H]),self.H*0.1])),
                                               int(max([min([y_pos + y_speed, self.W - 0.1*self.W]),self.W*0.1])))
        self.state['cursorDis'] = cursorDis = np.array(cursorPos) - np.array(prevCursorPos)
        if 'rotate_cw' in agent_events and self.state['target']:
            self.state['target'].rotate(-self.rot_size)
            reward += REWARD_DICT['hold_block'] / self.n_blocks

        if 'rotate_ccw' in agent_events and self.state['target']:
            self.state['target'].rotate(self.rot_size)
            reward += REWARD_DICT['hold_block'] / self.n_blocks

        #Penalize border hugging:
        if cursorPos[1] == self.W - 0.1*self.W or cursorPos[1] == self.W*0.1 or \
           cursorPos[0] == self.H - 0.1*self.H or cursorPos[0] == self.H*0.1:
            reward += REWARD_DICT['boundary'] / self.n_blocks
            penalize= True
        if self.state['grab']:
            #cursorDis = self.state['cursorDis']
            if self.state['target'] is None:
                for block in self.state['bList']:
                    if isinstance(block,PolyBlock):
                        boundary= block.size/2
                    elif isinstance(block,Disk):
                        boundary= block.size
                    if (prevCursorPos[0]>=(block.center[0]- boundary) and
                        prevCursorPos[0]<=(block.center[0]+ boundary) and
                        prevCursorPos[1]>=(block.center[1]- boundary) and
                        prevCursorPos[1]<=(block.center[1]+ boundary) ): # inside the bounding box

                        self.state['target']=block # "pick up" block
                        self.state['bList'].append(self.state['bList'].pop(self.state['bList'].index(block)))

            if self.state['target'] is not None:
                self.state['target'].center = tuple(np.array(self.state['target'].center) + cursorDis)

                if not penalize:
                    #reward += 0.1 / self.n_blocks
                    reward += REWARD_DICT['hold_block'] / self.n_blocks

        else:
            if self.state['target'] is not None:
                dists_and_holes = [(np.linalg.norm(np.array(self.state['target'].center)
                                                   - np.array(hole.center)),
                                    hole
                                    ) for hole in self.state['hList']]
                hole = min(dists_and_holes)[1]
                if fit(hole, self.state['target']):
                    self.state['bList'].remove(self.state['target'])
                    reward += REWARD_DICT['fit_block'] / self.n_blocks

            self.state['target'] = None

        for item in self.state['hList'] + self.state['bList']:
            item.render(self.screen, angle=5.0) # Draw all items

        #Render Cursor
        if self.state['grab']:
            col= BLUE
        else:
            col= GREEN
        pg.draw.circle(self.screen, col, self.state['cursorPos'], self.cursor_size)

        # All blocks have been removed.
        if self.state['bList'] == []:
            done= True
            reward+= REWARD_DICT['trial_end'] / self.n_blocks

        observation = process_observation(self.screen,self.H,self.rHW)

        return observation, reward, done, info

    def reset(self):
        self.initialize()
        observation, _, _, _ = self.step(0)
        return observation

    def render(self):
        pg.draw.rect(self.screen, BLACK, (self.H*0.1, self.W*0.1,
                                              self.H - 2*self.H*0.1, self.W - 2*self.W*0.1), 1)
        time.sleep(0.1)
        pg.display.flip()

def random_play(**kwargs):
    pass

def human_play(env_params, smooth= False):
    """
    Exectues the "main loop" for the shapesorting game, where actions are read
    from the keyboard:

    COMMANDS:
        Up, Down, Left, and Right arrow: move the cursor
        a, d: rotate counter clockwise and clockwise
        s: take a screenshot
        Space: Grab block.

    Args:
        env_params: keyword arguments for ShapeSorter. Should be unpacked directly
        from a SHAPESORT_ARGS dictionary from game_settings.py

        smooth: boolean indicating if actions should be "repeated", allowing
        the user to hold the movement and rotation keys. Makes human play more
        enjoyable, but less faithful to agent play
    """
    ss= ShapeSorter(**env_params)
    acts_taken = 0
    running = True
    actions= []
    while running:
        ss.reset()
        done= False
        for t in range(T):
            if smooth is False:
                actions= []
            flag = False
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running=False
                    break
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:
                        actions.append('grab')
                    #Adjust speed of cursor.
                    if event.key == pg.K_LEFT:
                        actions.append('left')
                    elif event.key == pg.K_RIGHT:
                        actions.append('right')
                    elif event.key == pg.K_UP:
                        actions.append('up')
                    elif event.key == pg.K_DOWN:
                        actions.append('down')
                    acts_taken += 1
                    flag= True
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_a:
                        actions.append('rotate_ccw')
                    elif event.key == pg.K_d:
                        actions.append('rotate_cw')

                    # saves screenshot in an h5 file.
                    elif event.key == pg.K_s:
                        with h5py.File("./plotting/screenshots.h5","a") as hf:
                            img = pg.surfarray.pixels3d(ss.screen)
                            hf.create_dataset("{}".format(np.random.randint(10)),data=img)

                if event.type == pg.KEYUP and smooth:
                    if event.key == pg.K_SPACE:
                        actions.remove('grab')
                    #Adjust speed of cursor.
                    if event.key == pg.K_LEFT:
                        actions.remove('left')
                    elif event.key == pg.K_RIGHT:
                        actions.remove('right')
                    elif event.key == pg.K_UP:
                        actions.remove('up')
                    elif event.key == pg.K_DOWN:
                        actions.remove('down')
                    elif event.key == pg.K_a:
                        actions.remove('rotate_ccw')
                    elif event.key == pg.K_d:
                        actions.remove('rotate_cw')

            if actions == []:
                actions.append('none')

            _,reward,done,info = ss.step(actions)
            ss.render()

            if done:
                break

def random_play(env_params, episodes= 100, timesteps= 50):
    """
    Exectues the "main loop" for the shapesorting game, where actions are
    selected randomly from the environment's action space.

    Args:
        env_params: keyword arguments for ShapeSorter. Should be unpacked directly
        from a SHAPESORT_ARGS dictionary from game_settings.py

        episodes: integer number of episodes
        timesteps: integer number of timesteps per episode

    """
    env = ShapeSorter(**env_params)
    for e in range(episodes):
        env.reset()
        for t in range(timesteps):
            a = env.action_space.sample()
            o, r, done, info = env.step(a)
            env.render()
            if done:
                break

if __name__ == '__main__':
    from game_settings import SHAPESORT_ARGS0
    #X = human_play(smooth= True, **SHAPESORT_ARGS0) # Execute our main function
    random_play(SHAPESORT_ARGS0)
