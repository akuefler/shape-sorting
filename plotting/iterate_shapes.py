import pygame as pg
import matplotlib.pyplot as plt
from shape_zoo import *
from plotting import *
from config import *
from util import Saver
import argparse

import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--data_time", type=str, default="17-01-19-22-35-45-287050")
#parser.add_argument("--data_times", nargs="+", type=str, default=["17-01-19-22-34-07-174659",
                                                                  #"17-01-19-22-34-56-056347",
                                                                  #"17-01-19-22-35-45-287050"])
args = parser.parse_args()

data_saver = Saver(time=args.data_time,path='{}/{}'.format(DATADIR,'scene_and_enco_data'))
data = data_saver.load_dictionary(0,"data")

X = data['X']
N = X.shape[0]

x = X[100,:,:,0]
# inferno is good ...
#plt.imshow(x, cmap=plt.cm.inferno)
#plt.show()

#cmaps = [plt.cm.inferno, plt.cm.magma, plt.cm.gist_stern,
         #plt.cm.nipy_spectral, plt.cm.CMRmap]
#f, axs = plt.subplots(1,len(cmaps))

#for cm, ax in zip(cmaps, axs):
    #ax.imshow(x,cmap=cm)
    
#plt.show()

with h5py.File("screenshots.h5","r") as hf:
    img0 = hf['0'][...]
    img1 = hf['4'][...]


W = 750
H = 200

screen = pg.display.set_mode((W,H))
screen.fill(WHITE)

shape_cls = np.array([Trapezoid, RightTri, Hexagon, Tri, Rect])
shape_cls = shape_cls[SHAPE_ORDER]
x = np.linspace(0,W,len(shape_cls) + 2)[1:-1]
shape_obj = [sh(BLACK, [pos, H/2], 100, "hole", angle=0.)
             for pos, sh in zip(x,shape_cls)]

for i, shape in enumerate(shape_obj):
    r = 180
    if i == 0:
        r = 225
    shape.rotate(r)    
    shape.render(screen,angle=0.)

pg.draw.rect(screen, BLACK, (W*0.05, H*0.1, W - 2*W*0.05, H - 2*H*0.1), 1)
X = pg.surfarray.pixels_red(screen).T
#f, ax = plt.subplots()
#ax.imshow(1. - X,cmap=plt.cm.Greys)

###
#ax.set_xticks([])
#ax.set_yticks([])
#ax.set_xticklabels([])
#ax.set_yticklabels([])
ax1 = plt.subplot2grid((2,2), (0,0), colspan=1)
ax2 = plt.subplot2grid((2,2), (0,1), colspan=1)
ax3 = plt.subplot2grid((2,2), (1, 0), colspan=2)

ax1.imshow(img0)
ax2.imshow(img1)
ax3.imshow(1. - X,cmap=plt.cm.Greys)

axs = [ax1, ax2, ax3]
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])    

plt.tight_layout()
plt.savefig("{}shapes.pdf".format(FIGDIR),bbox_inches='tight',format='pdf',dpi=300)

halt= True