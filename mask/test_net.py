import torch, sys, os, gc, random
import numpy as np 
from copy import deepcopy
import torch.nn as nn
import classify, utils, facenet

device = "cuda"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_name = "VGG16"

def load_state_dict(self, state_dict):
	own_state = self.state_dict()
	
	for name, param in state_dict.items():
		if name == 'module.fc_layer.weight':
			own_state['module.fc_layer.0.weight'].copy_(param.data)
		elif name == 'module.fc_layer.bias':
			own_state['module.fc_layer.0.bias'].copy_(param.data)
		elif name in own_state:
			own_state[name].copy_(param.data)
		else:
			print(name)

def test_net(net, test_loader, mode):
	net.eval()
	sum_acc = 0.0
	for r in range(10):
		torch.manual_seed(r) # cpu
		torch.cuda.manual_seed(r) #gpu
		np.random.seed(r) #numpy
		random.seed(r)
		ACC_cnt = utils.AverageMeter()
		for i, (img, one_hot, iden) in enumerate(testloader):
			img, one_hot, iden = img.to(device), one_hot.to(device), iden.to(device)
			bs = img.size(0)
			iden = iden.view(-1)
			
			___, out_prob, __ = net(img)
			
			out_iden = torch.argmax(out_prob, dim=1).view(-1)
			ACC = torch.sum(out_iden == iden).item() / bs
			ACC_cnt.update(ACC, bs)
		sum_acc += ACC_cnt.avg * 100
		print(ACC_cnt.avg)
		
	print("Test ACC:{:.2f}".format(sum_acc / 10))

if __name__ == "__main__":
	dataset_name = "celeba"
	file = "./config/" + dataset_name + ".json"
	args = utils.load_params(json_file=file)
	mode = "reg"
	
	if model_name.startswith("VGG16"):
		if model_name.startswith("VGG16_vib"):
			net = classify.VGG16_vib(args["dataset"]["num_of_classes"])
			mode = "vib"
		else:
			net = classify.VGG16(args["dataset"]["num_of_classes"])
		net = torch.nn.DataParallel(net).cuda()
	elif model_name.startswith("FaceNet64"):
		net = facenet.FaceNet64(args["dataset"]["num_of_classes"])
		net = torch.nn.DataParallel(net).cuda()
	elif model_name.startswith('FaceNet'):
		net = facenet.FaceNet(args["dataset"]["num_of_classes"])
		net = torch.nn.DataParallel(net).cuda()
	elif model_name.startswith('Lenet'):
		net = classify.Lenet(args["dataset"]["num_of_classes"])
		net = torch.nn.DataParallel(net).cuda()

	ckp_path = "./attack_model/" + model_name + ".tar"
	test_file_path = "/home/yhzhang/workspace/dataset/celeba/faceset1000/test_list.txt"
	testset, testloader = utils.init_dataloader(args, test_file_path, mode="inpainting")

	ckp = torch.load(ckp_path)['state_dict']
	load_state_dict(net, ckp)

	test_net(net, testloader, mode)

	
