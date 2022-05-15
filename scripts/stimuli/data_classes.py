from PIL import Image
import os
import numpy as np
import torch
from torch import from_numpy
from torch.utils.data import Dataset

class DewindDataSet(Dataset):
        def __init__(self, pic_dir,transform):
                self.pic_dir = pic_dir
                self.pic_names = sorted(os.listdir(self.pic_dir))
                self.transform=transform

        def __len__(self):
                return len(self.pic_names)

        def unpack_name(self,img_name):
                numerosity,size,spacing,num_lines,_ = img_name.split('_')
                return from_numpy(np.array(int(numerosity))), from_numpy(np.array(int(size))), from_numpy(np.array(int(spacing))), from_numpy(np.array(int(num_lines)))

        def __getitem__(self, idx):
                img_name = os.path.join(self.pic_dir,self.pic_names[idx])

                image = Image.open(img_name).convert('RGB')
                image = self.transform(image)
                
                numerosity, size, spacing, num_lines = self.unpack_name(self.pic_names[idx])
                sample = (image, numerosity, size, spacing, num_lines)
                return sample
