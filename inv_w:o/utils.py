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

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()

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
        tvls.save_image(images, filename, normalize = normalize, nrow=nrow, padding=0)


def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    #print(state_dict)
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        #print(param.data.shape)
        own_state[name].copy_(param.data)

def gen_hole_area(size, mask_size):
    """
    * inputs:
        - size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of hole area.
        - mask_size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of input mask.
    * returns:
            A sequence used for the input argument 'hole_area' for function 'gen_input_mask'.
    """
    mask_w, mask_h = mask_size
    harea_w, harea_h = size
    offset_x = random.randint(0, mask_w - harea_w)
    offset_y = random.randint(0, mask_h - harea_h)
    return ((offset_x, offset_y), (harea_w, harea_h))


def crop(x, area):
    """
    * inputs:
        - x (torch.Tensor, required)
                A torch tensor of shape (N, C, H, W) is assumed.
        - area (sequence, required)
                A sequence of length 2 ((X, Y), (W, H)) is assumed.
                sequence[0] (X, Y) is the left corner of an area to be cropped.
                sequence[1] (W, H) is its width and height.
    * returns:
            A torch tensor of shape (N, C, H, W) cropped in the specified area.
    """
    xmin, ymin = area[0]
    w, h = area[1]
    return x[:, :, ymin : ymin + h, xmin : xmin + w]

def get_center_mask(img_size, bs):
    mask = torch.zeros(img_size, img_size).cuda()
    scale = 0.15
    l = int(img_size * scale)
    u = int(img_size * (1.0 - scale))
    mask[l:u, l:u] = 1
    mask = mask.expand(bs, 1, img_size, img_size)
    return mask

def get_train_mask(img_size, bs):
    mask = torch.zeros(img_size, img_size).cuda()
    typ = random.randint(0, 1)
    if typ == 0:
        scale = 0.25
        l = int(img_size * scale)
        u = int(img_size * (1.0 - scale))
        mask[l:, l:u] = 1
    else:
        u, d = 10, 52
        l, r = 25, 40
        mask[l:r, u:d] = 0
        u, d = 26, 38
        l, r = 40, 63
        mask[l:r, u:d] = 0

    mask = mask.repeat(bs, 1, 1, 1)
    return mask

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
    batch = []
    for _ in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        x = torch.unsqueeze(dataset[index], dim=0)
        batch.append(x)
    return torch.cat(batch, dim=0)

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

def calc_feat(img):
    I = IR_50((112, 112))
    BACKBONE_RESUME_ROOT = "./feature/ir50.pth"
    print("Loading Backbone Checkpoint ")
    I.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
    I = torch.nn.DataParallel(I).cuda()
    img = low2high(img)
    feat = I(img)
    return feat

def get_inv_mask(args, img_size, bs):
    mask = torch.ones(img_size, img_size).to(device).float()
    if args["inpainting"]["masktype"] == 'center':
        scale = 0.25
        l = int(img_size * scale)
        u = int(img_size * (1.0 - scale))
        mask[l:u, l:u] = 0
    elif args["inpainting"]["masktype"] == 'eye':
        u, d = 10, 52
        l, r = 25, 40
        mask[l:r, u:d] = 0
    elif args["inpainting"]["masktype"] == 'face':
        u, d = 10, 52
        l, r = 25, 40
        mask[l:r, u:d] = 0
        u, d = 26, 38
        l, r = 40, 63
        mask[l:r, u:d] = 0

    elif args["inpainting"]["masktype"] == "all":
        mask[:, :] = 0
    
    elif args["inpainting"]["masktype"] == "big":
        scale = 0.25
        l = int(img_size * scale)
        u = int(img_size * (1.0 - scale))
        mask[l:, l:u] = 0
    #weight = utility.get_weight_matrix(mask, device)
    #weight = weight.to(device).float()
    weight = createWeightedMask(mask.cpu().numpy())
    weight = torch.from_numpy(weight).to(device).float()
    mask = mask.repeat(bs, 3, 1, 1)
    weight = weight.repeat(bs, 3, 1, 1)
    return mask, weight

def get_mask(args, img_size, bs):
    mask = torch.zeros(img_size, img_size).to(device).float()
    if args["inpainting"]["masktype"] == 'center':
        scale = 0.25
        l = int(img_size * scale)
        u = int(img_size * (1.0 - scale))
        mask[l:u, l:u] = 1
   
    elif args["inpainting"]["masktype"] == 'eye':
        u, d = 10, 52
        l, r = 25, 40
        mask[l:r, u:d] = 1
    
    elif args["inpainting"]["masktype"] == 'face':
        u, d = 10, 52
        l, r = 25, 40
        mask[l:r, u:d] = 1
        u, d = 26, 38
        l, r = 40, 63
        mask[l:r, u:d] = 1
    
    elif args["inpainting"]["masktype"] == "all":
        mask[:, :] = 1
    
    elif args["inpainting"]["masktype"] == "big":
        scale = 0.25
        l = int(img_size * scale)
        u = int(img_size * (1.0 - scale))
        mask[l:, l:u] = 1
        
    mask = mask.repeat(bs, 1, 1, 1)
   
    return mask

def createWeightedMask(mask, nsize = 7):
    """Takes binary weighted mask to create weighted mask as described in 
    paper.
    Arguments:
        mask - binary mask input. numpy float32 array
        nsize - pixel neighbourhood size. default = 7
    """
    ker = np.ones((nsize, nsize), dtype = np.float32)
    ker = ker / np.sum(ker)
    wmask = mask * convolve2d(mask, ker, mode = 'same', boundary = 'symm')
    return wmask

def get_model(model_name, classes):
    #classes = 1000
    if model_name.startswith('VGG16_vib'):
        net = classify.VGG16_vib(classes)
    elif model_name.startswith('VGG16_sen'):
        net = classify.VGG16_sen(classes)
    elif model_name.startswith('VGG16'):
        net = classify.VGG16(classes)
    elif model_name.startswith('Lenet'):
        net = classify.Lenet(classes)
    elif model_name.startswith('Simple_CNN'):
        net = classify.Simple_CNN(classes)
    elif model_name == "FaceNet64":
        net = facenet.FaceNet64(classes)
    elif model_name == 'Softmax':
        net = classify.Softmax(classes)
    return net

'''
def poisson_blend(x, output, mask):
    """
    * inputs:
        - x (torch.Tensor, required)
                Input image tensor of shape (N, 3, H, W).
        - output (torch.Tensor, required)
                Output tensor from Completion Network of shape (N, 3, H, W).
        - mask (torch.Tensor, required)
                Input mask tensor of shape (N, 1, H, W).
    * returns:
                An image tensor of shape (N, 3, H, W) inpainted
                using poisson image editing method.
    """
    x = x.clone().cpu()
    output = output.clone().cpu()
    mask = mask.clone().cpu()
    mask = torch.cat((mask,mask,mask), dim=1) # convert to 3-channel format
    num_samples = x.shape[0]
    ret = []
    for i in range(num_samples):
        dstimg = transforms.functional.to_pil_image(x[i])
        dstimg = np.array(dstimg)[:, :, [2, 1, 0]]
        srcimg = transforms.functional.to_pil_image(output[i])
        srcimg = np.array(srcimg)[:, :, [2, 1, 0]]
        msk = transforms.functional.to_pil_image(mask[i])
        msk = np.array(msk)[:, :, [2, 1, 0]]
        # compute mask's center
        xs, ys = [], []
        for i in range(msk.shape[0]):
            for j in range(msk.shape[1]):
                if msk[i,j,0] == 255:
                    ys.append(i)
                    xs.append(j)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        center = ((xmax + xmin) // 2, (ymax + ymin) // 2)
        out = cv2.seamlessClone(srcimg, dstimg, msk, center, cv2.NORMAL_CLONE)
        out = out[:, :, [2, 1, 0]]
        out = transforms.functional.to_tensor(out)
        out = torch.unsqueeze(out, dim=0)
        ret.append(out)
    ret = torch.cat(ret, dim=0)
    return ret
'''


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

def calc_acc(net, img, iden):
    #img = (img - 0.5) / 0.5
    img = low2high(img)
    __, ___, out_iden = net(img)
    iden = iden.view(-1, 1)
    bs = iden.size(0)
    acc = torch.sum(iden == out_iden).item() * 1.0 / bs
    return acc

def calc_center(feat, iden, path="feature"):
    iden = iden.long()
    feat = feat.cpu()
    center = torch.from_numpy(np.load(os.path.join(path, "center.npy"))).float()
    bs = feat.size(0)
    true_feat = torch.zeros(feat.size()).float()
    for i in range(bs):
        real_iden = iden[i].item()
        true_feat[i, :] = center[real_iden, :]
    dist = torch.sum((feat - true_feat) ** 2) / bs
    return dist.item()
    
def calc_knn(feat, iden, path="feature"):
    iden = iden.cpu().long()
    feat = feat.cpu()
    feats = torch.from_numpy(np.load(os.path.join(path, "feat.npy"))).float()
    info = torch.from_numpy(np.load(os.path.join(path, "info.npy"))).view(-1).long()
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


