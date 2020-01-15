import torch.nn.init as init
import h5py, os, facenet, sys
import json, PIL, time, random, torch, math
import dataloader, classify
import numpy as np
import pandas as pd
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvls
from torchvision import transforms
from datetime import datetime
from scipy.signal import convolve2d

device = "cuda"

class Timer(object):
    """Timer class."""
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self.start = time.time()

    def stop(self):
        self.end = time.time()
        self.interval = self.end - self.start

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def weights_init(m):
    if isinstance(m, model.MyConvo2d): 
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

def init_dataloader(args, file_path, batch_size=64, mode="gan"):
    with Timer() as t:
        if args['dataset']['name'] == "celeba":
            data_set = dataloader.ImageFolder(args, file_path, mode)
        else:
            data_set = dataloader.GrayFolder(args, file_path, mode)
            
        data_loader = torch.utils.data.DataLoader(data_set,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    pin_memory=True)
        
    print('Initializing data loader took %ds' % t.interval)
    return data_set, data_loader

def init_pubfig(args, img_path, batch_size, mode = "gan"):
    with Timer() as t:
        data_set = dataloader.PubImage(args, img_path, mode)
        data_loader = torch.utils.data.DataLoader(data_set,
                                    batch_size = batch_size,
                                    shuffle = False,
                                    num_workers = 4,
                                    pin_memory = True)
        
    print('Initializing data loader took %ds' % t.interval)
    return data_set, data_loader

def load_params(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

def print_params(info, params, dataset=None):
    print('-----------------------------------------------------------------')
    if dataset is not None:
        print("Dataset: %s" % dataset)
        print("Running time: %s" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    for i, (key, value) in enumerate(info.items()):
        if i >= 3: 
            print("%s: %s" % (key, str(value)))
    for i, (key, value) in enumerate(params.items()):
        print("%s: %s" % (key, str(value)))
    print('-----------------------------------------------------------------')

def save_tensor_images(images, filename, nrow = None, normalize = True):
    if not nrow:
        tvls.save_image(images, filename, normalize = normalize, padding=0)
    else:
        tvls.save_image(images, filename, normalize = normalize, nrow = nrow)


def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    #print(state_dict)
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        #print(param.data.shape)
        own_state[name].copy_(param.data)

def sample_random_batch(dataset, batch_size=32):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the input dataset.
    """
    num_samples = len(dataset)
    blur_list, gt_list = [], []
    for _ in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        blur, gt = dataset[index]
        blur = torch.unsqueeze(blur, dim=0)
        gt = torch.unsqueeze(gt, dim=0)
        blur_list.append(blur)
        gt_list.append(gt)
    return torch.cat(blur_list, dim=0), torch.cat(gt_list, dim=0)

def get_deprocessor():
    # resize 112,112
    proc = []
    proc.append(transforms.Resize((112, 112)))
    proc.append(transforms.ToTensor())
    return transforms.Compose(proc)

def low2high(img):
    # 0 and 1, 64 to 112
    bs = img.size(0)
    proc = get_deprocessor()
    img_tensor = img.detach().cpu().float()
    img = torch.zeros(bs, 3, 112, 112)
    for i in range(bs):
        img_i = transforms.ToPILImage()(img_tensor[i, :, :, :]).convert('RGB')
        img_i = proc(img_i)
        img[i, :, :, :] = img_i[:, :, :]
    
    img = img.cuda()
    return img

def calc_psnr(img1, img2):
    bs, c, h, w = img1.size()
    ten = torch.tensor(10).float().cuda()
    mse = (img1 - img2) ** 2
    # [bs, c, h, w]
    mse = torch.sum(mse, dim = 1)
    mse = torch.sum(mse, dim = 1)
    mse = torch.sum(mse, dim = 1).view(-1, 1) / (c * h * w)
    maxI = torch.ones(bs, 1).cuda()
    psnr = 20 * torch.log(maxI / torch.sqrt(mse)) / torch.log(ten)
    return torch.mean(psnr)

def calc_center(feat, iden):
    center = torch.from_numpy(np.load("feature/center.npy")).cuda()
    bs = feat.size(0)
    true_feat = torch.zeros(feat.size()).cuda()
    for i in range(bs):
        real_iden = iden[i].item()
        true_feat[i, :] = center[real_iden, :]
    dist = torch.sum((feat - true_feat) ** 2) / bs
    return dist.item()
    
def calc_knn(feat, iden):
    feats = torch.from_numpy(np.load("feature/feat.npy")).cuda()
    info = torch.from_numpy(np.load("feature/info.npy")).view(-1).long().cuda()
    bs = feat.size(0)
    tot = feats.size(0)
    knn_dist = 0
    for i in range(bs):
        knn = 1e8
        for j in range(tot):
            if info[j] == iden[i]:
                dist = torch.sum((feat[i, :] - feats[j, :]) ** 2)
                knn = min(knn, dist)
        knn_dist += knn
    return knn_dist / bs


