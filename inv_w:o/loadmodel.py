from generator import *
from discri import *
from losses import completion_network_loss
from utils import *
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss, DataParallel
from torchvision.utils import save_image
from torch.autograd import grad
from PIL import Image
import torchvision.transforms as transforms
import torch
import time
import random
import os
import argparse
import numpy as np
import json
import time

ld_input_size, cn_input_size = 32, 64

def load_model(path, net):
    checkpoint = torch.load(path)
    load_my_state_dict(net, checkpoint['state_dict'])
    print(net)
    


def test_model(test_set, net, iter_times = 0):
    mpv = torch.FloatTensor([0.5, 0.4, 0.4]).cuda().view(-1, 3, 1, 1)
    z_dim = 100
    result_img_dir = "result/inpaint_images"
    os.makedirs(result_img_dir, exist_ok=True)
    with torch.no_grad():
        
        x = sample_random_batch(test_set, batch_size=32).to(device)
        
        img_size, bs = x.size(2), x.size(0)
        mask = get_input_mask(img_size, bs)
        x_mask = x - x * mask + mpv * mask
        inp = torch.cat((x_mask, mask), dim=1)
        z1 = torch.randn(bs, z_dim).cuda()
        output1 = net(inp)
        #z2 = torch.randn(bs, z_dim).cuda()
        #output2 = net((inp, z2))
        #feat1, feat2 = calc_feat(output1), calc_feat(output2)
        #diff = torch.sum(torch.abs(feat1 - feat2)) / torch.sum(torch.abs(z1 - z2))
        #print("{:.2f}".format(diff))
        #completed = poisson_blend(x, output, mask)
        imgs = torch.cat((x.cpu(), x_mask.cpu(), output1.cpu()), dim=0)
        imgpath = os.path.join(result_img_dir, 'step%d.png' % iter_times)
        save_tensor_images(imgs, imgpath, nrow=bs)



if __name__ == "__main__":
    Net = CompletionNetwork()
    Net = torch.nn.DataParallel(Net).cuda()
    test_img_path = "/home/yhzhang/workspace/dataset/celeba/facetest"
    dataset_name = "celeba"
    batch_size = 64
    file = "./config/" + dataset_name + ".json"
    args = load_params(json_file=file)
    test_set, test_loader = init_dataloader(args, test_img_path, batch_size)
    checkpoint_path = "./premodels/Context_G.tar"
    load_model(checkpoint_path, Net)
    
    
    test_model(test_set, Net)
    





    
