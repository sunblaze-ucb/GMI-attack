import torch, sys, os, time, random, losses
import numpy as np 
import torch.nn as nn
from utils import *
from classify import *
from facenet import *
from discri import DWGANGP
from gen import BlurNet


ld_input_size = 32

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = "cuda"

n_classes = 1000
iter_times = 2000
momentum = 0.9
lamda = 100
lr = 2e-2
bs = 100

output_dir = 'result'
result_imgdir = os.path.join(output_dir, "deblur_images")
os.makedirs(result_imgdir, exist_ok=True)

def noise_loss(V, img1, img2):
    feat1, out1, ___ = V(img1)
    feat2, out2, ___ = V(img2)
    
    loss = torch.mean(torch.abs(feat1 - feat2))
    #return mse_loss(output * mask, input * mask)
    return loss

def load_state_dict(self, state_dict):
	own_state = self.state_dict()
	
	for name, param in state_dict.items():
		if name == 'module.fc_layer.0.weight':
			own_state['module.fc_layer.weight'].copy_(param.data)
		elif name == 'module.fc_layer.0.bias':
			own_state['module.fc_layer.bias'].copy_(param.data)
		elif name in own_state:
			own_state[name].copy_(param.data)
		else:
			print(name)


if __name__ == '__main__':

	dataset_name = "celeba"
	file = "./config/" + dataset_name + ".json"
	args = load_params(json_file=file)

	criteria = losses.CrossEntropyLoss().cuda()
	file_path = args['dataset']['test_file_path']
	
	if dataset_name == "celeba":
		data_set, data_loader = init_dataloader(args, file_path, bs, mode="attack")
	
	root_path = "result_model"
	attack_name = "FaceNet64_set2"

	print("Attack Model Name:" + attack_name)
	print("Iter times:{}".format(iter_times))
	print("Lamda:{:.2f}".format(lamda))
	print("LR:{:.2f}".format(lr))

	Net = BlurNet()
	Net = torch.nn.DataParallel(Net).cuda()

	D = DWGANGP()
	D = torch.nn.DataParallel(D).cuda()

	if attack_name.startswith("VGG16"):
		V = VGG16(1000)
	elif attack_name.startswith("IR152"):
		V = IR152(1000)
	elif attack_name.startswith("FaceNet64"):
	 	V = FaceNet64(1000)
	else:
		print("Model name Error")
		exit()

	V = torch.nn.DataParallel(V).cuda()
	
	path_V = "attack_model/" + attack_name + ".tar"
	path_N = os.path.join(root_path, "Blur_m1_s2.tar")
	path_D = os.path.join(root_path, "D_m1_s2.tar")
	
	ckp_N = torch.load(path_N)
	ckp_D = torch.load(path_D)
	ckp_V = torch.load(path_V)
	
	load_my_state_dict(Net, ckp_N['state_dict'])
	load_my_state_dict(D, ckp_D['state_dict'])
	load_state_dict(V, ckp_V['state_dict'])

	D.eval()
	Net.eval()
	V.eval()
	

	cnt = 0

	diff = AverageMeter()

	

		z1 = torch.randn(bs, 100).to(device).float()
		z2 = torch.randn(bs, 100).to(device).float()

		output1 = Net((blur_img, z1))
		output2 = Net((blur_img, z2))

		diff_loss = noise_loss(V, output1, output2)
		diff_loss = diff_loss / torch.mean(torch.abs(z2 - z1))
		diff.update(diff_loss.item(), bs)

		if cnt >= 100:
			break

	print("Diver:{:.2f}".format(diff.avg))


	
	
