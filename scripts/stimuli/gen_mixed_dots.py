### Master Dataset Generator ###
'''

This dataset generator combines various dot image styles into 
a single generator. Every output image is named as follows;

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


### FUNCTIONS ###

#Pick a number from a range of 0 to max integer, based on a gaussian distribution around the middle number
def gaussian_choice(num_range):
    x = np.arange(1,num_range)
    if num_range%2 == 0:
        y = np.arange(np.floor(num_range/2)+1-num_range, np.floor(num_range/2))
    else:
        y = np.arange(np.floor(num_range/2)+1-num_range, np.floor(num_range/2))
    yU, yL = y + 0.5, y - 0.5 
    prob = ss.norm.cdf(yU, scale = 2) - ss.norm.cdf(yL, scale = 2)
    prob = prob / prob.sum() #normalize the probabilities so their sum is 1
    return np.random.choice(x, p = prob)

def gen_radii_areacontroldiff(average_radius,num_dots, min_radius,max_radius):
    average_area = round(np.pi*average_radius**2,1)
    total_area = average_area*num_dots
    if num_dots == 1:
        return {1:average_radius}
    else:
        radii = {}
        num_below = gaussian_choice(num_dots)
        extra_area = 0
        for i in range(num_below):
            radii[num_dots-i] = round(np.random.uniform(min_radius,average_radius),1)
            extra_area += average_area - round(np.pi*radii[num_dots-i]**2,1)
        for i in range(1,num_dots-num_below+1):
            added_area = round(np.random.uniform(0,extra_area),1)
            radii[i] = round(np.sqrt((average_area+added_area)/np.pi),1)
            extra_area -= added_area
        return radii

def dot_size_position_generator(style, num_dots, pic_dim, max_dots, dot_dist,
                                max_radius, min_radius):
    average_radius = round(np.sqrt(max_dots*(min_radius+3)**2/num_dots),1)
    retry = True
    while retry:
        retry = False
        sizes = {}

        if style == 'totalareacontroldifferent':
            tacd_radii = gen_radii_areacontroldiff(average_radius,num_dots,min_radius,max_radius)
            total_area = 0 
            for key in tacd_radii:
                total_area += np.pi*tacd_radii[key]**2
    
        for i in range(1,num_dots+1):
            #get spatial position
            touching = True
            attempts = 0
            while touching:
                attempts += 1
                if style == 'random':
                    r = round(np.random.uniform(min_radius,max_radius),1)
                elif style == 'dotareacontrol':
                    r = min_radius + 2
                elif style == 'totalareacontrolsame':
                    r = average_radius
                elif style == 'totalareacontroldifferent':
                    r = tacd_radii[i]
                x = round(np.random.uniform(r,pic_dim-r),1)
                y = round(np.random.uniform(r,pic_dim-r),1)
                touching = False
                for dot in sizes:
                    distance = np.sqrt((x-sizes[dot][0])**2+(y-sizes[dot][1])**2)
                    if distance <= r+sizes[dot][2]+dot_dist:
                        if attempts >= 200:
                            retry = True
                            break
                        touching = True
                        break
            if retry:
                break
            sizes[i] = [x,y,r]      
    return sizes

def dot_color_generator(style,num_dots, color_dist):
    colors = {}
    pythag_color_dist = np.sqrt(color_dist**2*3)

    if style == 'bow':
        colors['background'] = (255,255,255)
        for dot_num in range(1,num_dots+1):
            colors[dot_num] = (0,0,0)
        return colors

    elif style == 'wob':
        colors['background'] = (0,0,0)
        for dot_num in range(1,num_dots+1):
            colors[dot_num] = (255,255,255)
        return colors       

    elif style == 'greyrandom':
        background_color = random.randint(0,255)
        colors['background'] = (background_color,background_color,background_color)
        for dot_num in range(1,num_dots+1):
            camo = True
            while camo:
                c = random.randint(0,255)
                if abs(c-background_color) > color_dist:
                    camo = False
            colors[dot_num] = (c,c,c)

    elif style == 'greysingle':
        background_color = random.randint(0,255)
        colors['background'] = (background_color,background_color,background_color) 
        camo = True
        while camo:
            c = random.randint(0,255)
            if abs(c-background_color) > color_dist:
                camo = False
        for dot_num in range(1,num_dots+1):
            colors[dot_num] = (c,c,c)

    elif style == 'colorrandom':
        colors['background'] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        for dot_num in range(1,num_dots+1):
            camo = True
            while camo:
                c = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                if np.sqrt((c[0]-colors['background'][0])**2+(c[1]-colors['background'][1])**2+(c[2]-colors['background'][2])**2) > pythag_color_dist:
                    camo = False
            colors[dot_num] = c

    elif style =='colorsingle':
        colors['background'] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        camo = True
        while camo:
            c = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            if np.sqrt((c[0]-colors['background'][0])**2+(c[1]-colors['background'][1])**2+(c[2]-colors['background'][2])**2) > pythag_color_dist:
                camo = False
        for dot_num in range(1,num_dots+1):
            colors[dot_num] = c
    return colors





def gen_images(args):
    for num_dots in range(args.min_dots,args.max_dots+1):
        print(num_dots)
        for pic_index in range(1,args.num_pics_per_category+1):
            img_file_name = '%s_%s_%s_%s.png'%(num_dots,args.color_style,args.condition,pic_index)
            toprint = img_file_name

            colors = dot_color_generator(args.color_style,num_dots, args.color_dist)
            img = Image.new('RGB', (args.pic_dim, args.pic_dim), color = colors['background'])

            if num_dots > 0:
                sizes = dot_size_position_generator(args.condition,num_dots,args.pic_dim,args.max_dots,args.dot_dist,args.max_radius,args.min_radius)
                for dot_num in range(1,num_dots+1):
                    toprint += '    '+str(sizes[dot_num])+str(colors[dot_num])
                    corners = [sizes[dot_num][0]-sizes[dot_num][2],sizes[dot_num][1]-sizes[dot_num][2],sizes[dot_num][0]+sizes[dot_num][2],sizes[dot_num][1]+sizes[dot_num][2]]
                    dotdraw = ImageDraw.Draw(img)
                    fill_color = colors['background'] if args.hollow else colors[dot_num]
                    dotdraw.ellipse(corners, fill=(fill_color), outline=(colors[dot_num]))


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
    parser.add_argument('--hollow', action='store_true', default=False,
                        help='Make the dots hollow (ie their middle is the same color as the background)')


    

    args = parser.parse_args()
    # reconcile arguments
    prefix = 'enumeration_'
    if args.hollow:
        prefix += 'hollow_'
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

