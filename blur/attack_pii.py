import torch, sys, os, time, random, losses, utils
import numpy as np 
import torch.nn as nn
from classify import *
from facenet import *
from discri import DWGANGP
from gen import BlurNet


ld_input_size = 32

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = "cuda"

n_classes = 1000
momentum = 0.9
lamda = 100
lr = 2e-2
bs = 100

output_dir = 'result_imgs'
os.makedirs(output_dir, exist_ok=True)

def test_net(net, test_loader):
	net.eval()
	ACC_cnt = AverageMeter()
	for i, (blur, img, one_hot, iden) in enumerate(test_loader):
		img, one_hot, iden = img.to(device), one_hot.to(device), iden.to(device)
		bs = img.size(0)
		iden = iden.view(-1, 1)
		
		___, out_prob, out_iden = net(img)
		out_iden = out_iden.view(-1, 1)
		ACC = torch.sum(out_iden == iden).item() / bs
		ACC_cnt.update(ACC, bs)
		
	print("Test ACC:{:.2f}".format(ACC_cnt.avg * 100))

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
	args = utils.load_params(json_file=file)

	eval_id = 1

	criteria = losses.CrossEntropyLoss().cuda()
	file_path = args['dataset']['test_file_path']

	if dataset_name == "celeba":
		data_set, data_loader = utils.init_dataloader(args, file_path, bs, mode="attack")

	root_path = "result_model"

	Net = BlurNet()
	Net = torch.nn.DataParallel(Net).cuda()

	D = DWGANGP()
	D = torch.nn.DataParallel(D).cuda()

	path_N = os.path.join(root_path, "Blur_m1.tar")
	path_D = os.path.join(root_path, "D_m1.tar")

	ckp_N = torch.load(path_N)
	ckp_D = torch.load(path_D)

	utils.load_my_state_dict(Net, ckp_N['state_dict'])
	utils.load_my_state_dict(D, ckp_D['state_dict'])

	D.eval()
	Net.eval()

	cnt = 0
	tf = time.time()

	for blur_img, real_img, one_hot, iden in data_loader:
		
		
		blur_img, real_img, one_hot, iden = blur_img.to(device), real_img.to(device), one_hot.to(device), iden.to(device)
		iden = iden.view(-1)

		z = torch.randn(bs, 100).to(device).float()
		output = Net((blur_img, z))
		
		bs = blur_img.size(0)
		imgs = torch.cat((blur_img, output, real_img), dim=0)
		imgpath = os.path.join(output_dir, 'pii_{}.png'.format(cnt))
		utils.save_tensor_images(imgs, imgpath, nrow=bs)

		

	
	
