import os, utils, torchvision
import json, PIL, time, random
import torch, math, cv2

import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F 
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from sklearn.model_selection import train_test_split

class ImageFolder(data.Dataset):
    def __init__(self, args, file_path, mode):
        self.args = args
        self.mode = mode
        self.img_path = args["dataset"]["img_path"]
        self.blur_path = args['dataset']['blur_path']
        self.model_name = args["dataset"]["model_name"]
        self.img_list = os.listdir(self.img_path)
        self.processor = self.get_processor()
        self.blur_processor = self.blur_processor()
        self.name_list, self.label_list, self.attr_list = self.get_list(file_path) 
        self.image_list, self.blur_list = self.load_img()
        self.num_img = len(self.image_list)
        self.n_classes = args["dataset"]["n_classes"]
        print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path):
        name_list, label_list, attr_list = [], [], []
        f = open(file_path, "r")
        for line in f.readlines():
            line = line.strip().split(' ')
            img_name, iden = line[0], int(line[1])
            name_list.append(img_name)
            label_list.append(int(iden))
            line = line[2:]
            out = []
            for attr in line:
                out.append(int(attr))

            out = np.array(out)
            attr_list.append(np.array(out))

        return name_list, label_list, attr_list

    def load_img(self):
        img_list, blur_list = [], []
        for i, img_name in enumerate(self.name_list):
            if img_name.endswith(".png"):
                path = os.path.join(self.img_path, img_name)
                blur_path = os.path.join(self.blur_path, img_name)
                img = PIL.Image.open(path)
                img = img.convert('RGB')
                blur_img = PIL.Image.open(blur_path)
                blur_img = blur_img.convert('RGB')
                img_list.append(img)
                blur_list.append(blur_img)
        return img_list, blur_list
    
    def get_processor(self):
        if self.model_name == "FaceNet":
            re_size = 112
        else:
            re_size = 64
            
        crop_size = 108
        
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

        proc = []
        proc.append(transforms.ToTensor())
        proc.append(transforms.Lambda(crop))
        proc.append(transforms.ToPILImage())
        proc.append(transforms.Resize((re_size, re_size)))
        proc.append(transforms.ToTensor())
            
        return transforms.Compose(proc)

    def blur_processor(self):
        proc = [transforms.ToTensor()]
        return transforms.Compose(proc)

    def __getitem__(self, index):
        img = self.processor(self.image_list[index])
        blur_img = self.blur_processor(self.blur_list[index])
        if self.mode == "blur":
            return blur_img, img
        label = self.label_list[index]
        one_hot = np.zeros(self.n_classes)
        one_hot[label] = 1
        attr = self.attr_list[index]
        if self.mode == "attr":
            return blur_img, img, one_hot, attr
        return blur_img, img, one_hot, label

    def __len__(self):
        return self.num_img




    

