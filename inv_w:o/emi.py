import torch, os, time, random, losses
import torchvision.utils as tvls
import utils, classify, facenet
import torch.nn as nn
import numpy as np 

lr = 0.01
momentum = 0.9
iter_times = 1500

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
device = "cuda"

target = 3


if __name__ == '__main__':
	
	bs = 100
	n_classes = 1000
	one_hot, iden = torch.zeros(bs, n_classes), torch.zeros(bs)
	for i in range(bs):
		one_hot[i, i] = 1
		iden[i] = i
	
	iden = iden.view(-1).float().cuda()
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
	ckp_V = torch.load(path_V)

	V = torch.nn.DataParallel(V).cuda()

	utils.load_my_state_dict(V, ckp_V['state_dict'])

	print("Model Initalize")
	E = facenet.FaceAtt()
	E = torch.nn.DataParallel(E).cuda()
	path_E = "eval_model/" + eval_name + ".tar"

	utils.load_my_state_dict(G, ckp_G)
	utils.load_my_state_dict(D, ckp_D)
	utils.load_my_state_dict(T, ckp_T)
	utils.load_my_state_dict(E, ckp_E)
	
	criteria = losses.CrossEntropyLoss().cuda()
	
	V.eval()
	
	max_cnt, max_cnt_5 = 0, 0

	flag = torch.zeros(100)
	max_score = torch.zeros(100)
	max_iden = torch.zeros(100)
	
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
		feat = F(utils.low2high(fake))
		center, knn = utils.calc_center(feat, iden), utils.calc_knn(feat, iden)

		fake_img = z.detach()
		utils.save_tensor_images(fake, "emi_celeba.png", nrow=8)

		interval = time.time() - tf
		print("Time:{:.2f}\tCenter:{:.2f}\tKnn:{:.2f}".format(interval, center, knn))

	correct = 0
	for i in range(bs):
		if max_iden[i].item() == i:
			correct += 1
	
	correct_5 = torch.sum(flag)
	acc, acc_5 = correct * 1.0 / bs, correct_5 * 1.0 / bs
	print("Acc:{:.3f}\tAcc5:{:.3f}".format(acc, acc_5))

	
		
		
	

