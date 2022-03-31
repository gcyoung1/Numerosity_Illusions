### Dot comparison generator ###
'''

This generator makes images of the solitaire illusion;
outputs;
[# of dots]tot_[# blue dots]b_[# red dots]r_[unique image #].png

algorithm keeps the total area of each dot color the same, while keeping dot size pseudo random

Adjust and read descriptions for parameters below
'''


from PIL import Image, ImageDraw
import random
import numpy as np
import os
import pdb
import time
from copy import deepcopy





### PARAMETERS (Set these) ###

pic_dim = 256        # pixel dimension of one axis of output image, must be larger than ((2*max_radius)+dot_dist)*num_rows + 2*max_radius
max_radius = 15     # maximum dot size      
min_radius = 5       # minimum dot size
max_dot_dist = 15     # maximum spacing b/w dots
min_dot_dist = 2       # minimum spacing b/w dots
num_rows = 6        #Number of dots for each color
group_sizes = [1,2,3]          #Number of dots in group for each image type
num_pics_per_category = 20      #number of pictures per image category
#distribution_type = 'normal'    # way dot sizes are distributed, can be 'normal' or 'uniform' 


outputdir = '../../stimuli/test_solitaire/'
if not os.path.exists(outputdir):
    os.mkdir(outputdir)

#COLORS
RED = (255,0,0)     #color for blue
BLUE = (0,0,255)      #color for red
WHITE = (255,255,255)
BLACK = (0,0,0)
background_color = BLACK

### STOP EDITING ###

#seed
np.random.seed(123)
random.seed(123)

### FUNCTIONS ###

def getCorners(row,col,radius,dot_dist,origin):
  xorigin=origin[0]
  yorigin=origin[1]
  x0 = xorigin+col*(2*radius+dot_dist)
  y0 = yorigin+row*(2*radius+dot_dist)
  x1=x0+2*radius
  y1=y0+2*radius
  return(x0,y0,x1,y1)


def getMaxRadius(pic_dim, num_rows, min_dot_dist):
    return (pic_dim-(num_rows-1)*min_dot_dist)/(2*(num_rows))

def getMaxDotDist(pic_dim, num_rows, radius):
    return (pic_dim - 2*radius*(num_rows))/(num_rows-1)

def maxOriginCoord(pic_dim,radius,dot_dist,numDots):
    return pic_dim-((numDots-1)*(2*radius+dot_dist)+2*radius)


### MAIN SCRIPT ###



#Main Image Generation Loop
start_time = time.time()

max_radius=min(max_radius, getMaxRadius(pic_dim, num_rows, min_dot_dist))

#pdb.set_trace()
for pic in range(1,num_pics_per_category+1):

    radius = random.uniform(min_radius,max_radius)
    max_dot_dist = min(getMaxDotDist(pic_dim, num_rows, radius), max_dot_dist)
    dot_dist = random.uniform(min_dot_dist,max_dot_dist)
    maxOriginX = maxOriginCoord(pic_dim,radius,dot_dist,3)
    maxOriginY = maxOriginCoord(pic_dim,radius,dot_dist,num_rows)
    originX = random.uniform(0,maxOriginX)
    originY = random.uniform(0,maxOriginY)
    origin = [originX,originY]
    cols = [0,1,2]
    
    for solidCol in [0,1]:
    
        together = "differentsides" if solidCol else "sameside"
        brokenCols=list(set(cols)-set([solidCol]))
        
        for switch_every in group_sizes:

            folder_name = '%srows_%sswitch_%s_%s'%(num_rows,switch_every,together,pic)
            blue_solid_file_name = '%srows_%sswitch_%s_%s_%s.png'%(num_rows,switch_every,together,pic,'blue')
            red_solid_file_name = '%srows_%sswitch_%s_%s_%s.png'%(num_rows,switch_every,together,pic,'red')
    
            blue_solid = Image.new(mode="RGB", size=(pic_dim,pic_dim),color=(background_color))
            red_solid = Image.new(mode="RGB", size=(pic_dim,pic_dim),color=(background_color))
            
            blue_draw = ImageDraw.Draw(blue_solid)
            red_draw = ImageDraw.Draw(red_solid)
            
            brokenColIdx = 0
            for y in range(num_rows):    
              corners = getCorners(y, solidCol,radius,dot_dist,origin)
              blue_draw.pieslice(corners, start=0,end=360,fill=(BLUE), outline=(BLUE))
              red_draw.pieslice(corners, start=0,end=360,fill=(RED), outline=(RED))
            
              corners = getCorners(y, brokenCols[brokenColIdx],radius,dot_dist,origin)
              blue_draw.pieslice(corners,start=0,end=360, fill=(RED), outline=(RED))
              red_draw.pieslice(corners,start=0,end=360, fill=(BLUE), outline=(BLUE))
            
              if (y+1)%switch_every==0:
                brokenColIdx = not (brokenColIdx and brokenColIdx)
            
            savePath = os.path.join(outputdir,folder_name)
            os.mkdir(savePath)
            blue_solid.save(os.path.join(savePath,blue_solid_file_name))
            red_solid.save(os.path.join(savePath,red_solid_file_name))


end_time = time.time()
print('Run Time: %s'%(end_time-start_time))

