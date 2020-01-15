import os
import time
import torch
import dataloader
import torchvision
from utils import *
import PIL.Image as Image
from torch.nn import BCELoss
from torch.autograd import grad
import torchvision.utils as tvls
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from discri import DiscriminatorWGANGP, DGWGAN, DGWGAN32
from generator import Generator, GeneratorMNIST, GeneratorCXR

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False) 

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)

def gradient_penalty(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs = torch.ones(o.size()).cuda(), create_graph = True)[0].view(z.size(0), -1)
    gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gp

""" param """
epochs = 200
batch_size = 64
n_critic = 5
lr = 0.0002
z_dim = 100
dataset_name = "celeba"

if __name__ == "__main__":
    
    file = "./config/" + dataset_name + ".json"
    args = load_params(json_file=file)
    save_imgdir = "result/imgs_celeba_gan"
    save_modeldir= "result/models_celeba_gan"
    os.makedirs(save_modeldir, exist_ok=True)
    os.makedirs(save_imgdir, exist_ok=True)
    file_path = args['dataset']['train_file_path']
    dataset, dataloader = init_dataloader(args, file_path, batch_size, mode="gan")

    if dataset_name == "celeba":
        G = Generator(z_dim)
        DG = DiscriminatorWGANGP(3)
    elif dataset_name == "MNIST":
        G = GeneratorMNIST(z_dim)
        DG = DGWGAN32(in_dim=1)
    elif dataset_name == "cxr":
        G = GeneratorCXR(z_dim)
        DG = DGWGAN(in_dim=1)
    else:
        print("Dataset does not exist")
        exit()

    print("Dataset:{}".format(dataset_name))

    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))


    for epoch in range(epochs):
        start = time.time()
        for i, imgs in enumerate(dataloader):
            step = epoch * len(dataloader) + i + 1
            imgs = imgs.cuda()
            bs = imgs.size(0)
            
            freeze(G)
            unfreeze(DG)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            r_logit = DG(imgs)
            f_logit = DG(f_imgs)
            
            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            gp = gradient_penalty(imgs.data, f_imgs.data)
            dg_loss = - wd + gp * 10.0
            
            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            # train G

            if step % n_critic == 0:
                freeze(DG)
                unfreeze(G)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                logit_dg = DG(f_imgs)
                # calculate g_loss
                g_loss = - logit_dg.mean()
                
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start
        print("Epoch:%d \t Time:%.2f" % (epoch, interval))
        if (epoch+1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(), os.path.join(save_imgdir, "result_image_{}.png".format(epoch)), nrow = 8)
        
        torch.save({'state_dict':G.state_dict()}, os.path.join(save_modeldir, "celeba_G.tar"))
        torch.save({'state_dict':DG.state_dict()}, os.path.join(save_modeldir, "celeba_D.tar"))

