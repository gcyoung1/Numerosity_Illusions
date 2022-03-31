'''

This dataset generator generates binary images of white squares on black backgrounds. There are three orthogonal dimensions: field area (the size of the area the squares are in), square side length, and number of squares in the image.

File names of images, placed in the args.dataset_name directory of the /stimuli/ directory, follow the following format:
[# of dots]_[square side length]_[field area side length]_[unique image #].png

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


def gen_origin_in_range(bounding_origin, bounding_side, side):
    x0 = np.random.randint(bounding_origin[0],bounding_origin[0]+bounding_side-side+1)
    y0 = np.random.randint(bounding_origin[1],bounding_origin[1]+bounding_side-side+1)
    return (x0,y0)
    
def intersects(origin1,origin2,min_distance):
    (x0,y0) = origin1
    (x1,y1) = origin2
    x_distance = abs(x0-x1)
    y_distance = abs(y0-y1)
    return (x_distance < min_distance) and (y_distance < min_distance)


def gen_square_origins(num_squares, square_side, pic_dim, square_spacing, bounding_origin, bounding_side):
    retry = True
    while retry:
        retry = False
        square_origins = []
        
        for i in range(1,num_squares+1):
            #get spatial position
            touching = True
            (x0,y0) = gen_origin_in_range(bounding_origin, bounding_side, square_side)
            attempts = 0
            while touching:
                attempts += 1
                touching = False
                for square_origin in square_origins:
                    if intersects((x0,y0),square_origin,square_spacing + square_side):
                        if attempts >= 200:
                            retry = True
                            break
                        touching = True
                        break
            if retry:
                break
            square_origins.append((x0,y0))
    return square_origins



def gen_images(args):

    for num_squares in range(args.min_squares,args.max_squares+1):
        print(num_squares)
        for pic_index in range(1,args.num_pics_per_category+1):
            img_file_name = f"{num_squares}_{args.square_side}_{args.bounding_side}_{pic_index}.png"
            toprint = img_file_name
            img = Image.new('1', (args.pic_dim, args.pic_dim), 'black')
            bounding_origin = gen_origin_in_range((0,0),args.pic_dim,args.bounding_side)

            if num_squares > 0:
                square_origins = gen_square_origins(num_squares, args.square_side, args.pic_dim, args.square_spacing, bounding_origin, args.bounding_side)
                for square_origin in square_origins:
                    toprint += '    '+str(square_origin)
                    corners = [square_origin[0],square_origin[1],square_origin[0]+square_side-1,square_origin[1]+square_side-1]#-1 bc annoyingly a (0,0,0,0) rectangle in PIL is a 1x1 rectangle
                    squaredraw = ImageDraw.Draw(img)
                    squaredraw.rectangle(corners, fill='white',outline='white')
            
            if pic_index <= args.num_train_pics_per_category:
                img.save(os.path.join(args.train,img_file_name))
            else:
                img.save(os.path.join(args.test,img_file_name))
            for i in range(args.max_squares-num_squares):
                toprint+='    '
            args.outputfile.write(toprint+'\n')
            args.outputfile.flush()





if __name__ == '__main__':

    start_time = time.time()

        #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Generate Dewind stimuli')
    parser.add_argument('--dataset-name', type=str, default='test_dewind', metavar='',
                        help='name for dataset folder in /stimuli/')
    parser.add_argument('--pic-dim', type=int, default=100, metavar='',
                        help='number of pixels for each axis of image')
    parser.add_argument('--square-sides', nargs='+', default=[], metavar='I',type=int,
                        help='space separated list of the number of pixels for each axis of squares')
    parser.add_argument('--bounding-sides', nargs='+', default=[], metavar='I',type=int,
                        help='space separated list of the number of pixels for each axis of field area')
    parser.add_argument('--square-spacing', type=int, default=3, metavar='',
                        help='minimum number of pixels between squares')
    parser.add_argument('--num-squares', nargs='+', default=[], metavar='I',type=int,
                        help='min and mx number of squares to generate separated by a space')
    parser.add_argument('--num-pics-per-category', type=int, default=10, metavar='',
                        help='number of pictures per category')
    parser.add_argument('--num-train-pics-per-category', type=int, default=10, metavar='',
                        help='number of training pictures per category')
    

    args = parser.parse_args()
    # reconcile arguments
    args.dataset_name = 'dewind_'+args.dataset_name
    if not os.path.exists(os.path.join('../../stimuli',args.dataset_name)):
        os.mkdir(os.path.join('../../stimuli',args.dataset_name))
    args.outputdir = os.path.join('../../stimuli',args.dataset_name)
    os.mkdir(os.path.join(args.outputdir,'train'))
    args.train = os.path.join(args.outputdir,'train')
    os.mkdir(os.path.join(args.outputdir,'test'))
    args.test = os.path.join(args.outputdir,'test')
    args.outputfile = open(os.path.join(args.outputdir,'img_stats.tsv'),'w+')

    args.min_squares = args.num_squares[0]
    args.max_squares = args.num_squares[-1]
    
    print('running with args:')
    print(args)

    

    #Setup header for output image stats file
    column_nums = range(1,args.max_squares+1)
    column_names = []
    for i in column_nums:
        column_names.append('square'+str(i))
    args.outputfile.write("image name    %s\n"%'    '.join(column_names))
    args.outputfile.flush()


    for square_side in args.square_sides:
        args.square_side = square_side
        print(f"Square side: {square_side}")
        for bounding_side in args.bounding_sides:
            args.bounding_side = bounding_side
            print(f"Bounding side: {bounding_side}")
            gen_images(args)

    end_time = time.time()
    print('Run Time: %s'%(end_time-start_time))

