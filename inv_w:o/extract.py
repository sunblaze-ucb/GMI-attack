import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as tvls


import numpy as np
import os
import utils
from facenet import FaceNet, IR_50_pre

def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

def extract_feature(args, backbone, input_size = [112, 112], rgb_mean = [0.5, 0.5, 0.5], rgb_std = [0.5, 0.5, 0.5], embedding_size = 512, device = "cuda"):
    crop_size = 108
    re_size = 64
    offset_height = (218 - crop_size) // 2
    offset_width = (178 - crop_size) // 2
    crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(crop),
        transforms.ToPILImage(),
        transforms.Resize([112, 112]),  
        transforms.ToTensor()])
    
    bs = args["inpainting"]["batch_size"]
    img_path = args["dataset"]["test_img_path"]
    dataset, dataloader = utils.init_dataloader(args, img_path, bs, transform, mode="inpainting")

    # extract features
    backbone.eval()
    
    
    idx = 0
    
    
    tot = 19036
    features = torch.zeros(tot, embedding_size).cuda()
    info = torch.zeros(tot, 1).cuda()
    with torch.no_grad():
        for img, one_hot, iden in dataloader:
            img, one_hot, iden = img.to(device), one_hot.to(device), iden.to(device)
            feat =  backbone(img)
            bs = img.size(0)
            for i in range(bs):
                real_iden = iden[i].item()
                features[idx, :] = feat[i, :]
                info[idx] = real_iden
                idx += 1
                

    print(idx)
    return features, info
    
    
    '''
    features = torch.zeros(1000, embedding_size).cuda()
    num = torch.zeros(1000, 1).cuda()
    with torch.no_grad():
        for img, one_hot, iden in dataloader:
            img, one_hot, iden = img.to(device), one_hot.to(device), iden.to(device)
            feat =  backbone(img)
            bs = img.size(0)
            for i in range(bs):
                real_iden = iden[i].item()
                features[real_iden, :] += feat[i, :]
                num[real_iden] += 1
                idx += 1
    '''   
    print(idx)
    return features, info
    
    

if __name__ == "__main__":
    dataset_name = "celeba"
    file = "./config/" + dataset_name + ".json"
    args = utils.load_params(json_file = file)
    


    BACKBONE = IR_50_pre((112, 112))
    BACKBONE_RESUME_ROOT = "premodels/ir50.pth"
    print("Loading Backbone Checkpoint ")
    BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
    I = torch.nn.DataParallel(BACKBONE).cuda()
    print("Model Initalize")

    feat, info = extract_feature(args, I)
    feat_np = feat.cpu().detach().numpy()
    info_np = info.cpu().detach().numpy()
    np.save("feature/feat.npy", feat_np)
    np.save("feature/info.npy", info_np)
    
    '''
    center = extract_feature(args, I)
    center_np = center.cpu().detach().numpy()
    np.save("feature/center.npy", center_np)
    '''
   

