import torch, os, time, random, losses
import torchvision.utils as tvls
import utils, classify, facenet
import torch.nn as nn
import numpy as np 

lr = 0.01
momentum = 0.9
iter_times = 1500
bs = 100
n_classes = 1000

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
device = "cuda"

target = 1

def load_attr_label():
	file_path = "/home/yhzhang/workspace/dataset/celeba/iden_attr.txt"
	f_iden = open(file_path, "r")
	attr_list = []
	for line in f_iden.readlines():
		line_list = line.split(' ')
		hmy = []
		for attr in line_list[1:]:
			hmy.append(int(attr))
		hmy = np.array(hmy).reshape(1, -1)
		attr_list.append(hmy)
	
	attr = np.concatenate(attr_list, axis=0)
	attr = torch.from_numpy(attr)
	return attr


if __name__ == '__main__':
	attr = load_attr_label()
	attr = attr[:bs, :].cuda()
	one_hot = torch.zeros(bs, n_classes)
	for i in range(bs):
		one_hot[i, i] = 1
	one_hot = one_hot.float().cuda()
	
	#model_name in ["Lenet", "Simple_CNN", "VGG16"]
	if target == 1:
		attack_name = "VGG16_set2"
	elif target == 2:
		attack_name = "IR152_set2"
	elif target == 3:
		attack_name = "FaceNet64_set2"
	
	eval_name = "FaceNet_att"

	print("Attack model name:" + attack_name)
	print("Iter times:{}".format(iter_times))
	
	if attack_name.startswith("Lenet"):
		V = classify.Lenet(n_classes)
	elif attack_name.startswith("VGG16"):
		V = classify.VGG16(n_classes)
	elif attack_name.startswith("IR152"):
		V = classify.IR152(n_classes)
	elif attack_name.startswith("FaceNet64"):
		V = facenet.FaceNet64(n_classes)
	else:
		print("Model doesn't exist")
		exit()

	path_V = "attack_model/" + attack_name + ".tar"
	path_E = "eval_model/" + eval_name + ".tar"
	ckp_V = torch.load(path_V)['state_dict']
	ckp_E = torch.load(path_E)['state_dict']

	V = torch.nn.DataParallel(V).cuda()

	print("Model Initalize")
	E = facenet.FaceAtt()
	E = torch.nn.DataParallel(E).cuda()
	

	utils.load_my_state_dict(V, ckp_V)
	utils.load_my_state_dict(E, ckp_E)
	
	criteria = losses.CrossEntropyLoss().cuda()
	
	V.eval()
	E.eval()
	
	for random_seed in range(1):

		tf = time.time()
		torch.manual_seed(random_seed) # cpu
		torch.cuda.manual_seed(random_seed) #gpu
		np.random.seed(random_seed) #numpy
		random.seed(random_seed)

		z = torch.zeros(bs, 3, 64, 64).to(device).float()
		z.requires_grad = True
		v = torch.zeros(bs, 3, 64, 64).to(device).float()

		for i in range(iter_times):
			fake = z.clone()
			__, out_prob, __ = V(fake)

			if z.grad is not None:
				z.grad.data.zero_()

			Iden_Loss = criteria(out_prob, one_hot) 
			Total_Loss = 100 * Iden_Loss

			Total_Loss.backward()
			
			v_prev = v.clone()
			gradient = z.grad.data
			v = momentum * v - lr * gradient
			z = z + ( - momentum * v_prev + (1 + momentum) * v)
			z = torch.clamp(z.detach(), 0, 1).float()
			z.requires_grad = True

		fake = z.clone()
		out = E(utils.low2high(fake))
		out_cls = (out >= 0.5)
		eq = (out_cls.long() == attr.long()) * 1.0
		acc = torch.mean(eq.float(), dim=0)
		out_str = ''
		for i in range(acc.shape[0]):
			out_str += "%.2f " %(acc[i].item())

		interval = time.time() - tf
		print("Time:{:.2f}\t".format(interval))
		print(out_str)


	

	
		
		
	

