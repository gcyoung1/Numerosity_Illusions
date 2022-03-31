### Master Dataset Generator ###
'''

This dataset generator is the same as gen_mixed_dots.py, only
it generates images with some number of pairs of dots connected by lines.
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
from itertools import combinations

### FUNCTIONS ###
def distance_point_to_line(p3, line):
    p1, p2 = line
    line = p2-p1
    p2_to_dot = p3-p2
    if (p2_to_dot).dot(line) > 0:
        return np.linalg.norm(p2_to_dot)
    p1_to_dot = p3-p1
    if (p1_to_dot).dot(line) < 0:
        return np.linalg.norm(p1_to_dot)
    return np.linalg.norm(np.cross(line, p1_to_dot))/np.linalg.norm(line)

#Taken from stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
# Return true if line segments AB and CD intersect
def lines_intersect(a,b):
    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    A,B = a[0], a[1]
    C,D = b[0], b[1]
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)    


def intersects_other_dots(dots, start_dot_key, end_dot_key, line, line_dist):
    for key, dot in dots.items():
        if key in [start_dot_key, end_dot_key]: continue
        dot_center = np.array(dot)[[0,1]]
        dot_radius = dot[2]
        if (dot_radius + line_dist) >= distance_point_to_line(dot_center, line):
            return True
    return False

def intersects_other_lines(lines, line):
    for existing_line in lines:
        if lines_intersect(line, existing_line):
            return True
    return False
        
def get_potential_lines(unconnected_dot_keys, dots):
    potential_lines = []
    for [start_dot_key, end_dot_key] in combinations(unconnected_dot_keys, r=2):
        line = [np.array(dots[start_dot_key])[[0,1]], np.array(dots[end_dot_key])[[0,1]]]
        line.sort(key = lambda x: x[1], reverse=True)
        dist = euclidean(line[0], line[1])
        potential_lines.append((dist, [start_dot_key, end_dot_key], line))
    
    potential_lines.sort(key = lambda line: line[0])
    return potential_lines


def illusory_line_generator(lines_to_draw, line_dist, dots, redo=0):
    unconnected_dot_keys = list(range(1,len(dots)+1))
    potential_lines = get_potential_lines(unconnected_dot_keys, dots)
    potential_lines = potential_lines[redo:]
    if len(potential_lines) < lines_to_draw:
        return False

    lines = []
    used_dot_keys = []
    for _ in range(lines_to_draw):
        line_valid = False
        while(not line_valid):
            line_valid = True
            if len(potential_lines) == 0:
                return illusory_line_generator(lines_to_draw, line_dist, dots, redo+1)
            next_shortest_line = potential_lines.pop(0)
            [start_dot_key, end_dot_key] = next_shortest_line[1]
            if start_dot_key in used_dot_keys or end_dot_key in used_dot_keys: 
                line_valid = False
                continue
            line = next_shortest_line[2]

            if intersects_other_dots(dots, start_dot_key, end_dot_key, line, line_dist):
                line_valid = False
                continue
            if intersects_other_lines(lines, line):
                line_valid = False
                continue
            
        used_dot_keys.extend(next_shortest_line[1])
        lines.append(line)
        
    return lines


def line_generator(lines_to_draw, line_length_range, line_dist, dots):
    unconnected_dot_keys = list(range(1,len(dots)+1))
    lines = []

    for _ in range(lines_to_draw):
        valid_line = False
        attempts = 0
        while not valid_line:
            if attempts > 500: return False
            attempts += 1
            valid_line = True
            [start_dot_key, end_dot_key] = np.random.choice(unconnected_dot_keys, 2, replace=False)
            line = [np.array(dots[start_dot_key])[[0,1]], np.array(dots[end_dot_key])[[0,1]]]

            #Line valid length
            valid_line_length = (line_length_range[0] < np.linalg.norm(line[0]-line[1]) < line_length_range[1])
            if not valid_line_length:
                line_valid = False
                continue
            if intersects_other_dots(dots, start_dot_key, end_dot_key, line, line_dist):
                line_valid = False
                continue
            if intersects_other_lines(lines, line):
                line_valid = False
                continue
            
        lines.append(line)
        unconnected_dot_keys = [x for x in unconnected_dot_keys if not x in [start_dot_key, end_dot_key]]

    return lines

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
                assert (num_dots - (2*args.connecting_lines)) >= 0, f"Can't make {2*args.connecting_lines} pairs from {num_dots} dots."

                if args.illusory:
                    lines = illusory_line_generator(args.connecting_lines, args.line_dist, sizes)
                    if not lines:
                        continue

                else:
                    lines = line_generator(args.connecting_lines, args.line_length_range, args.line_dist, sizes)
                    if not lines:
                        continue
                break

            for line in lines:
                linedraw = ImageDraw.Draw(img)
                line_color = colors['background'] if args.illusory else colors[1]
                linedraw.line([tuple(line[0]), tuple(line[1])], fill=(line_color), width=args.line_width)

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
    # parser.add_argument('--free-lines', type=int, metavar='',
    #                     help='number of free-hanging lines in the image')
    parser.add_argument('--illusory', action='store_true', default=False,
                        help='Make connecting lines the same color as the background (ie illusory contours)')
    parser.add_argument('--connecting-lines', type=int, metavar='',
                        help='number of lines which connect pairs of dots')
    parser.add_argument('--line-length-range', nargs='+', default=[2, 30], type=int,
                        help='minimum dot radius and maximum dot radius separated by a space')
    parser.add_argument('--line-dist', type=int, default=3,
                        help='minimum number of pixels between lines and dots')
    parser.add_argument('--line-width', type=int, default=1,
                        help='width of lines')
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
    if len(args.color_styles) > 1 or not 'colorsingle' in args.color_styles:
        args.color_styles = ['colorsingle']
        print('Changing color_styles to colorsingle')
    prefix = 'barbell_'
    if args.illusory:
        prefix += 'illusory_'
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

