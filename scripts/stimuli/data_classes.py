from PIL import Image
import os
import numpy as np
from torchvision import datasets, transforms, utils
np.set_printoptions(threshold=np.inf)
import torch
from torch import from_numpy
from torch.autograd import Variable
from torch.utils.data import Dataset
import re
import torch.nn as nn
import torch.nn.functional as F

default_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def gaussian_blur(img):
        kernel = [[0.003765,0.015019,0.023792,0.015019,0.003765],
        [0.015019,0.059912,0.094907,0.059912,0.015019],
        [0.023792,0.094907,0.150342,0.094907,0.023792],
        [0.015019,0.059912,0.094907,0.059912,0.015019],
        [0.003765,0.015019,0.023792,0.015019,0.003765]]
        kernel = nn.Parameter(torch.FloatTensor(kernel)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
                return F.conv2d(img.unsqueeze(0),kernel).squeeze(0)

dewind_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(gaussian_blur)
    ])



class CountingDotsDataSet(Dataset):
        """How many dots are in this pic dataset."""

        def __init__(self, root_dir, train=True,transform = default_transform):
                """
                """

                self.root_dir = root_dir
                if type(root_dir) is str:    
                        if train:
                                self.pic_dir = os.path.join(root_dir,'train')
                        else:
                                self.pic_dir = os.path.join(root_dir,'test')
                        self.pic_names = sorted(os.listdir(self.pic_dir))
                else:
                        self.pic_names = []
                        self.pic_dirs = []
                        for root in root_dir:
                                if train:
                                        pic_dir = os.path.join(root,'train')
                                else:
                                        pic_dir = os.path.join(root,'test')
                                self.pic_names.extend(sorted(os.listdir(pic_dir)))
                                for f in sorted(os.listdir(pic_dir)):
                                        self.pic_dirs.append(pic_dir)

                self.transform=transform
                self.train=train

        def __len__(self):
                return len(self.pic_names)

        def get_label_from_name(self,img_name):
                split_name = img_name.split('_')
                num = int(split_name[0])
                num_class = num
                return np.array(num_class)       

        def get_condition_from_name(self,img_name):
                split_name = img_name.split('_')
                return split_name[2]


        def __getitem__(self, idx):
                if type(self.root_dir) is str:
                        img_name = os.path.join(self.pic_dir,self.pic_names[idx])
                else:
                        img_name = os.path.join(self.pic_dirs[idx],self.pic_names[idx])

                image = Image.open(img_name)
                image = self.transform(image)


                label = self.get_label_from_name(self.pic_names[idx])
                label = from_numpy(label)
                if self.train:
                        sample = (image,label)
                else:
                        condition = self.get_condition_from_name(self.pic_names[idx])
                        sample = (image, label, condition)
                return sample

class ComparisonDotsDataSet(Dataset):
        """Which color has more dots."""
        # file name example   10tot_3b_7r_8.png
        def __init__(self, root_dir, train=True, transform=default_transform):
                """
                """
                self.root_dir = root_dir
                self.transform = transform
                self.train = train
                if type(root_dir) is str:    
                        if train:
                                self.pic_dir = os.path.join(root_dir,'train')
                        else:
                                self.pic_dir = os.path.join(root_dir,'test')
                        self.pic_names = sorted(os.listdir(self.pic_dir))
                else:       #feeding a list of multiple folders, should probably just get rid of this
                        self.pic_names = []
                        self.pic_dirs = []
                        for root in root_dir:
                                if train:
                                        pic_dir = os.path.join(root,'train')
                                else:
                                        pic_dir = os.path.join(root,'test')
                                self.pic_names.extend(sorted(os.listdir(pic_dir)))
                                for f in sorted(os.listdir(pic_dir)):
                                        self.pic_dirs.append(pic_dir)

        def __len__(self):
                return len(self.pic_names)

        def get_label_from_name(self,img_name):
                temp = re.findall(r'\d+', img_name)
                numbers = list(map(int, temp))
                tot = numbers[0]
                blue = numbers[1]
                red = numbers[2]
                ratio = float(blue)/float(red)
                index = numbers[3]
                if blue > red:
                        label = np.array(0)
                else:
                        label = np.array(1)
                # colorsingle = False
                # if 'r' not in img_name:
                #         colorsingle=True

                return label, tot, blue, red, ratio, index#, colorsingle
         

        def __getitem__(self, idx):
                if type(self.root_dir) is str:
                        img_name = os.path.join(self.pic_dir,self.pic_names[idx])
                else:
                        img_name = os.path.join(self.pic_dirs[idx],self.pic_names[idx])

                image = Image.open(img_name)
                image = self.transform(image)

                label, tot, num_blue, num_red,ratio,index = self.get_label_from_name(self.pic_names[idx])
                #label, tot, num_blue, num_red,ratio,index,colorsingle = self.get_label_from_name(self.pic_names[idx])
                label = from_numpy(label)
                if self.train:
                        sample = (image,label)
                else:
                        sample = (image,label, img_name, tot, num_blue, num_red, ratio, index)
                return sample

class EstimatingDotsDataSet(Dataset):
        """How many blue dots are there."""
        # file name example   10tot_3b_7r_8.png
        def __init__(self, root_dir, train=True, transform=default_transform):
                """
                """
                self.root_dir = root_dir
                self.transform = transform
                self.train = train
                if type(root_dir) is str:    
                        if train:
                                self.pic_dir = os.path.join(root_dir,'train')
                        else:
                                self.pic_dir = os.path.join(root_dir,'test')
                        self.pic_names = sorted(os.listdir(self.pic_dir))
                else:       #feeding a list of multiple folders, should probably just get rid of this
                        self.pic_names = []
                        self.pic_dirs = []
                        for root in root_dir:
                                if train:
                                        pic_dir = os.path.join(root,'train')
                                else:
                                        pic_dir = os.path.join(root,'test')
                                self.pic_names.extend(sorted(os.listdir(pic_dir)))
                                for f in sorted(os.listdir(pic_dir)):
                                        self.pic_dirs.append(pic_dir)

        def __len__(self):
                return len(self.pic_names)

        def get_label_from_name(self,img_name):
                temp = re.findall(r'\d+', img_name)
                numbers = list(map(int, temp))
                tot = numbers[0]
                blue = numbers[1]
                red = numbers[2]
                index = numbers[3]

                return np.array(float(blue))
         

        def __getitem__(self, idx):
                if type(self.root_dir) is str:
                        img_name = os.path.join(self.pic_dir,self.pic_names[idx])
                else:
                        img_name = os.path.join(self.pic_dirs[idx],self.pic_names[idx])

                image = Image.open(img_name)
                image = self.transform(image)

                num_blue = self.get_label_from_name(self.pic_names[idx])
                num_blue = from_numpy(num_blue)
                sample = (image,num_blue)
                return sample
                
                
                
                
                
                
class SolitaireDataSet(Dataset):
        """How many blue dots are there."""
        # file name example   12rows_3switch_0solid_1_red.png
        def __init__(self, root_dir, transform=default_transform):
                """
                """
                self.root_dir = root_dir
                self.transform = transform
                self.pic_dirs = os.listdir(self.root_dir)

        def __len__(self):
                return len(self.pic_dirs)

        def get_label_from_name(self,img_name):
                temp = re.findall(r'\d+', img_name)
                numbers = list(map(int, temp))
                num_rows = numbers[0]
                group_size = numbers[1]
                together = img_name.split('_')[2]
                index = numbers[2]
                return num_rows, group_size, together
         

        def __getitem__(self, idx):
                blue_solid_name = os.path.join(self.root_dir,self.pic_dirs[idx],self.pic_dirs[idx]+'_blue.png')
                red_solid_name = os.path.join(self.root_dir,self.pic_dirs[idx],self.pic_dirs[idx]+'_red.png')

                blue_image = Image.open(blue_solid_name)
                red_image = Image.open(red_solid_name)
                blue_image = self.transform(blue_image)
                red_image = self.transform(red_image)

                num_rows, group_size, together = self.get_label_from_name(self.pic_dirs[idx])
                sample = (red_image,blue_image,num_rows,group_size,together)
                return sample


class SymbolDataSet(Dataset):
        """What number is this."""
        # file name example   'number1_comicsans_13.png'
        def __init__(self, root_dir,train=True, transform=default_transform):

                self.root_dir = root_dir
                if train:
                        self.pic_dir = os.path.join(root_dir,'train')
                else:
                        self.pic_dir = os.path.join(root_dir,'test')
                
                self.pic_names = sorted(os.listdir(self.pic_dir))
                self.transform=transform
                self.train=train

        def __len__(self):
                return len(self.pic_names)

        def get_label_from_name(self,img_name):
                temp = re.findall(r'\d+', img_name)
                numbers = list(map(int, temp))
                num = numbers[0]
                num_class = num
                font = img_name.split('_')[1]
                return np.array(num_class),font       


        def __getitem__(self, idx):
                img_name = os.path.join(self.pic_dir,self.pic_names[idx])

                image = Image.open(img_name)
                image = self.transform(image)

                label,font = self.get_label_from_name(self.pic_names[idx])
                label = from_numpy(label)
                if self.train:
                        sample = (image,label)
                else:
                        sample=(image,label,font)
                return sample


class EnumerationAndSymbolsDataSet(Dataset):
        """Mix symbol and enumeration datasets."""
        def __init__(self, enumeration_root,symbol_root,train=True, transform=default_transform):
                """
                """
                self.enumeration_loader = CountingDotsDataSet(enumeration_root, train,transform)
                self.symbol_loader = SymbolDataSet(symbol_root, train,transform)

                self.enumeration_size = len(self.enumeration_loader)
                self.symbol_size = len(self.symbol_loader)

        def __len__(self):
                return self.enumeration_size  + self.symbol_size

        def __getitem__(self, idx):
                
                if idx<self.enumeration_size:
                        return self.enumeration_loader[idx]
                else:
                        return self.symbol_loader[idx-self.enumeration_size]


class DewindDataSet(Dataset):
        """How many dots are in this pic dataset."""

        def __init__(self, root_dir, train=True,transform = dewind_transform):
                """
                """

                self.root_dir = root_dir
                if train:
                        self.pic_dir = os.path.join(root_dir,'train')
                else:
                        self.pic_dir = os.path.join(root_dir,'test')

                self.pic_names = sorted(os.listdir(self.pic_dir))

                self.transform=transform
                self.train=train

        def __len__(self):
                return len(self.pic_names)

        def get_label_from_name(self,img_name):
                num,_,_,_ = img_name.split('_')
                return from_numpy(np.array(int(num)))

        def unpack_name(self,img_name):
                num,square_side,bounding_side,_ = img_name.split('_')
                return from_numpy(np.array(int(num))), from_numpy(np.array(int(square_side))), from_numpy(np.array(int(bounding_side))) 


        def __getitem__(self, idx):
                img_name = os.path.join(self.pic_dir,self.pic_names[idx])

                image = Image.open(img_name)
                image = self.transform(image)

                if self.train:
                        label = self.get_label_from_name(self.pic_names[idx])
                        sample = (image,label)
                else:
                        label, square_side, bounding_side = self.unpack_name(self.pic_names[idx])
                        sample = (image, label, square_side, bounding_side)
                return sample
