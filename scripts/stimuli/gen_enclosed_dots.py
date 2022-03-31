### Master Dataset Generator ###
'''

This dataset generator is the same as gen_mixed_dots.py, only
it generates images with some number of pairs of dots enclosed by ellipses.
The file format remains as below:

[# of dots]_[image color style]_[image size style]_[unique image #].png

Here is a description of the color style types generated;

bow:
   Black dots on a white background
wob:
   White dots on a black background
greyrandom:
   grey dots on random grey background with a random grey color assigned to each grey dot
greysingle:
    grey dots on random grey background with a single random grey color assigned to all dots
colorrandom:
   color dots on random color background with a random color assigned to each color dot
colorsingle:
    color dots on random color background with a single random color assigned to all dots

here is a description of the size style types generated;

random:
    each dot generated is a random size within the range
dotareacontrol:
    a single small dot size is used for every dot generated 
totalareacontrolsame:
    every image uses the same total dot area, and for each class (number of dots) the size of every
    dot is the same
totalareacontroldifferent:
    every image uses the same total dot area, the sizes of inidivual dots within each image still vary
    as determined by gen_dot_sizes_same_area()
'''


from PIL import Image, ImageDraw
import random
import numpy as np
import os
import scipy
import scipy.stats as ss
import time
import argparse
from gen_mixed_dots import gaussian_choice, gen_radii_areacontroldiff, dot_size_position_generator, dot_color_generator
from scipy.spatial.distance import euclidean
from gen_barbell_dots import distance_point_to_line, lines_intersect, get_potential_lines
from itertools import product
from ellipse import Ellipse
from geometry_utils import ccw_angle

### FUNCTIONS ###

def intersects_other_dots(dots, start_dot_key, end_dot_key, ellipse, ellipse_dot_dist):
    for key, dot in dots.items():
        if key in [start_dot_key, end_dot_key]: continue;
        dot_center = np.array(dot)[[0,1]]
        dot_radius = dot[2]
        [angle, _] = ccw_angle(ellipse.center, dot_center)
        if (ellipse.radius_at_angle(angle) + dot_radius + ellipse_dot_dist) >= euclidean(ellipse.center, dot_center):
            return True
    return False

def intersects_other_ellipses(ellipses, ellipse, ellipse_ellipse_dist):
  for existing_ellipse in ellipses:
    [ellipse_angle, existing_ellipse_angle] = ccw_angle(ellipse.center, existing_ellipse.center)
    if (ellipse.radius_at_angle(ellipse_angle) + existing_ellipse.radius_at_angle(existing_ellipse_angle) + ellipse_ellipse_dist) >= euclidean(ellipse.center, existing_ellipse.center):
      return True
  return False


def leaves_screen(pic_dim, ellipse):
  screen_edges = np.array([[[0,0], [0, pic_dim]],
                           [[0,pic_dim], [pic_dim, pic_dim]],
                           [[pic_dim,pic_dim], [pic_dim, 0]],
                           [[pic_dim,0], [0, 0]]])

  center = ellipse.transform_point(ellipse.center)

  for edge in screen_edges:
    line = ellipse.transform_point(edge)
    if distance_point_to_line(center, line) <= 1:#What about width?
      return True
  return False

def ellipse_generator(pic_dim, ellipses_to_draw, width, ellipse_dot_dist, ellipse_ellipse_dist, dots, redo=0):
    # if redo==0:
    #     print("")
    # print(f"Redo: {redo}")
    unconnected_dot_keys = list(range(1,len(dots)+1))
    potential_lines = get_potential_lines(unconnected_dot_keys, dots)
    potential_lines = potential_lines[redo:]
    if len(potential_lines) < ellipses_to_draw:
        return False

    ellipses = []
    used_dot_keys = []
    for ellipse_num in list(range(ellipses_to_draw)):
        ellipse_valid = False
        while(not ellipse_valid):
#            print(f"Potential lines left: {len(potential_lines)}")
            ellipse_valid = True
            if len(potential_lines) == 0:
                return ellipse_generator(pic_dim, ellipses_to_draw, width, ellipse_dot_dist, ellipse_ellipse_dist, dots, redo+1)
            next_shortest_line = potential_lines.pop(0)
            [start_dot_key, end_dot_key] = next_shortest_line[1]
            # if start_dot_key in used_dot_keys or end_dot_key in used_dot_keys: 
            #     ellipse_valid = False
            #     continue
            line = next_shortest_line[2]
            max_radius = max(dots[start_dot_key][2], dots[end_dot_key][2])
            ellipse = Ellipse(np.array(line), ellipse_dot_dist + max_radius, width)
            # print(f"\nProposed ellipse: ")
            # print(ellipse.stringify())

            if intersects_other_dots(dots, start_dot_key, end_dot_key, ellipse, ellipse_dot_dist):
#                print("Intersects dots")
                ellipse_valid = False
                continue
            if intersects_other_ellipses(ellipses, ellipse, ellipse_ellipse_dist):
#                print("Intersects ellipses")
                ellipse_valid = False
                continue
            if leaves_frame(pic_dim, ellipse):
#                print("Leaves frame")
                ellipse_valid = False
                continue

        used_dot_keys.extend(next_shortest_line[1])
        ellipses.append(ellipse)

    return ellipses

def gen_images(args):
    for num_dots in range(args.min_dots,args.max_dots+1):
        print(num_dots)
        for pic_index in range(1,args.num_pics_per_category+1):
            img_file_name = '%s_%s_%s_%s.png'%(num_dots,args.color_style,args.condition,pic_index)
            toprint = img_file_name

            colors = dot_color_generator(args.color_style,num_dots, args.color_dist)


            while (True):
                img = Image.new('RGB', (args.pic_dim, args.pic_dim), color = colors['background'])
                #Draw dots
                if num_dots > 0:
                    sizes = dot_size_position_generator(args.condition,num_dots,args.pic_dim,args.max_dots,args.dot_dist,args.max_radius,args.min_radius)
                    for dot_num in range(1,num_dots+1):
                        toprint += '    '+str(sizes[dot_num])+str(colors[dot_num])
                        corners = [sizes[dot_num][0]-sizes[dot_num][2],sizes[dot_num][1]-sizes[dot_num][2],sizes[dot_num][0]+sizes[dot_num][2],sizes[dot_num][1]+sizes[dot_num][2]]
                        dotdraw = ImageDraw.Draw(img)
                        dotdraw.ellipse(corners, fill=(colors[dot_num]), outline=(colors[dot_num]))

                #Draw lines
                assert (num_dots - (2*args.connecting_ellipses)) >= 0, f"Can't make {2*args.connecting_ellipses} pairs from {num_dots} dots."

                ellipses = ellipse_generator(args.pic_dim, args.connecting_ellipses, args.ellipse_width, args.ellipse_dot_dist, args.ellipse_ellipse_dist, sizes)
                if not ellipses:
                    continue
                break

            for ellipse in ellipses:
                ellipse_layer = Image.new('RGBA', (args.pic_dim, args.pic_dim), color = (colors[1][0],colors[1][1],colors[1][2],0))
                ellipsedraw = ImageDraw.Draw(ellipse_layer)
                corners = (ellipse.center[0]-ellipse.b, ellipse.center[1]-ellipse.a, ellipse.center[0]+ellipse.b, ellipse.center[1]+ellipse.a)
                ellipsedraw.ellipse(corners, fill=None, outline=(colors[1]), width = ellipse.width)
                center = (int(ellipse.center[0]), int(ellipse.center[1]))
                img.paste(ellipse_layer, (0,0), ellipse_layer.rotate(90 + ellipse.angle, center = center))

            if pic_index <= args.num_train_pics_per_category:
                img.save(os.path.join(args.train,img_file_name))
            else:
                img.save(os.path.join(args.test,img_file_name))




if __name__ == '__main__':

    start_time = time.time()

        #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Generate Dewind stimuli')
    parser.add_argument('--dataset-name', type=str, default='test',
                        help='name for dataset folder in /data/stimuli/')
    parser.add_argument('--pic-dim', type=int, default=256, 
                        help='number of pixels for each axis of image')
    parser.add_argument('--radius-range', nargs='+', default=[2, 30], type=int,
                        help='minimum dot radius and maximum dot radius separated by a space')
    parser.add_argument('--dot-dist', type=int, default=3,
                        help='minimum number of pixels between squares')
    parser.add_argument('--connecting-ellipses', type=int, metavar='',
                        help='number of ellipses which connect pairs of dots')
    parser.add_argument('--ellipse-width', type=int, default=2,
                        help='width of the edges of the ellipses')
    parser.add_argument('--ellipse-dot-dist', type=int, default=3,
                        help='minimum number of pixels between ellipses and dots')
    parser.add_argument('--ellipse-ellipse-dist', type=int, default=3,
                        help='minimum number of pixels between two ellipses')
    parser.add_argument('--color-dist', type=int, default=40,
                        help='minimum RGB distance between different colors in the image, background or between dots')
    parser.add_argument('--num-dots-range', nargs='+', default=[1,9],type=int,
                        help='min and mx number of dots to generate separated by a space')
    parser.add_argument('--num-pics-per-category', type=int, metavar='',
                        help='number of pictures per category')
    parser.add_argument('--num-train-pics-per-category', type=int, metavar='',
                        help='number of training pictures per category')
    parser.add_argument('--conditions', nargs='+',type=str, default=['random','totalareacontrolsame','totalareacontroldifferent', 'dotareacontrol'],
                        help='conditions to generate dots with')
    parser.add_argument('--color_styles', nargs='+',type=str, default=['colorsingle'],
                        help='conditions to generate dots with, choices are: wob, bow, greysingle, greyrandom, colorsingle, colorrandom')


    

    args = parser.parse_args()
    # reconcile arguments
    prefix = 'enclosed_'
    args.dataset_name = prefix+args.dataset_name

    if not os.path.exists(os.path.join('../../data/stimuli',args.dataset_name)):
        os.mkdir(os.path.join('../../data/stimuli',args.dataset_name))
    args.outputdir = os.path.join('../../data/stimuli',args.dataset_name)
    os.mkdir(os.path.join(args.outputdir,'train'))
    args.train = os.path.join(args.outputdir,'train')
    os.mkdir(os.path.join(args.outputdir,'test'))
    args.test = os.path.join(args.outputdir,'test')

    if len(args.num_dots_range) < 2:
        raise ValueError('args.num_dots_range must be two values')
    args.min_dots = args.num_dots_range[0]
    args.max_dots = args.num_dots_range[-1]

    if len(args.radius_range) < 2:
        raise ValueError('args.radius_range must be two values')
    args.min_radius = args.radius_range[0]
    args.max_radius = args.radius_range[-1]
    
    args_file = open(os.path.join(args.outputdir,'args.txt'),'a')
    args_file.write(str(args))
    args_file.close()
    print('running with args:')
    print(args)

    start_time = time.time()


    for condition in args.conditions:
        print(f"Condition: {condition}")
        args.condition = condition

        for color_style in args.color_styles:
            args.color_style = color_style
            print(f"Color Style: {color_style}")

            gen_images(args)



    end_time = time.time()
    print('Run Time: %s'%(end_time-start_time))

