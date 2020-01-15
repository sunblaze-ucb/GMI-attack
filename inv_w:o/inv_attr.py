import torch, os, time, random, generator, discri, classify, utils, facenet, losses
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls

device = "cuda"
dataset = "celeba"
n_classes = 1000
iter_times = 0
momentum = 0.9
lamda = 100
clip_range = 1
lr = 1e-2
bs = 100

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

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
	
	# model_name in ["VGG16", "IR152", "FaceNet64"]

	attack_name = "VGG16_set2"
	eval_name = "FaceNet_att"
	
	print("Attack model name:" + attack_name)
	print("Attack number:{}".format(bs))
	print("Eval name:" + eval_name)
	print("Iter times:{}".format(iter_times))
	print("Clip range:{}".format(clip_range))
	print("lamda:{}".format(lamda))
	print("LR:{:.2f}".format(lr))

	path_G = "result_model/celeba_G_v2.tar"
	path_D = "result_model/celeba_D_v2.tar"
	path_T = "attack_model/" + attack_name + ".tar"
	path_E = "eval_model/" + eval_name + ".tar"
	ckp_G = torch.load(path_G)['state_dict']
	ckp_D = torch.load(path_D)['state_dict']
	ckp_T = torch.load(path_T)['state_dict']
	ckp_E = torch.load(path_E)['state_dict']

	D = discri.DiscriminatorWGANGP(3)
	G = generator.Generator(100)	
	G = torch.nn.DataParallel(G).cuda()
	D = torch.nn.DataParallel(D).cuda()
	
	if attack_name.startswith("VGG16"):
		T = classify.VGG16(n_classes)
	elif attack_name.startswith("IR152"):
		T = classify.IR152(n_classes)
	elif attack_name.startswith("FaceNet64"):
		T = facenet.FaceNet64(n_classes)
	else:
		print("Model doesn't exist")
		exit()
	
	T = torch.nn.DataParallel(T).cuda()
	
	E = facenet.FaceAtt()
	E = torch.nn.DataParallel(E).cuda()

	utils.load_my_state_dict(G, ckp_G)
	utils.load_my_state_dict(D, ckp_D)
	utils.load_my_state_dict(T, ckp_T)
	utils.load_my_state_dict(E, ckp_E)
	
	closs = losses.CrossEntropyLoss().cuda()
	
	D.eval()
	G.eval()
	T.eval()
	E.eval()

	print("Model initialize finish!")
	print("Start inversion attack!")

	for random_seed in range(5):
		tf = time.time()
		
		torch.manual_seed(random_seed) # cpu
		torch.cuda.manual_seed(random_seed) #gpu
		np.random.seed(random_seed) #numpy
		random.seed(random_seed)

		z = torch.randn(bs, 100).to(device).float()
		z.requires_grad = True
		
		v = torch.zeros(bs, 100).to(device).float()
			
		best_ACC = 0.0

		for i in range(iter_times):
			fake = G(z)
			label = D(fake)
			__, out, ___ = T(fake)
			
			if z.grad is not None:
				z.grad.data.zero_()

			Prior_Loss = - label.mean()
			Iden_Loss = closs(out, one_hot)
			Total_Loss = Prior_Loss + lamda * Iden_Loss

			Total_Loss.backward()
			
			v_prev = v.clone()
			gradient = z.grad.data
			v = momentum * v - lr * gradient
			z = z + ( - momentum * v_prev + (1 + momentum) * v)
			z = torch.clamp(z.detach(), -clip_range, clip_range).float()
			z.requires_grad = True

			Prior_Loss_val = Prior_Loss.item()
			Iden_Loss_val = Iden_Loss.item()

			if (i+1) % 500 == 0:
				print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\t".format(i+1, Prior_Loss_val, Iden_Loss_val))
			
		fake = G(z)
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

	
		

	

