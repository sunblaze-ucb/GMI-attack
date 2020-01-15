import torch, os, time, random, generator, discri, classify, utils, facenet, losses
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls

device = "cuda"
dataset = "celeba"
n_classes = 1000
iter_times = 3000
momentum = 0.9
lamda = 100
lr = 2e-2
bs = 100


if __name__ == '__main__':
	
	bs = 100
	n_classes = 1000
	one_hot, iden = torch.zeros(bs, n_classes), torch.zeros(bs)
	for i in range(bs):
		one_hot[i, i] = 1
		iden[i] = i
	
	iden = iden.view(-1, 1).float().cuda()
	one_hot = one_hot.float().cuda()
	
	#model_name in ["IR50", "VGG16", "IR152"]
	
	attack_name = "IR50"
	eval_name = "FaceNet"
	print("Attack model name:" + attack_name)
	print("Eval name:" + eval_name)
	print("Iter times:{}".format(iter_times))
	print("lamda:{}".format(lamda))
	print("LR:{:.2f}".format(lr))

	path_T = "attack_model/" + attack_name + ".tar"
	path_E = "eval_model/" + eval_name + ".tar"
	ckp_T = torch.load(path_T)['state_dict']
	ckp_E = torch.load(path_E)['state_dict']

	if attack_name.startswith("Lenet"):
		T = classify.Lenet(n_classes)
	elif attack_name.startswith("VGG16"):
		T = classify.VGG16(n_classes)
	elif attack_name.startswith("IR50"):
		T = classify.IR50(n_classes)
	elif attack_name.startswith("IR152"):
		T = classify.IR152(n_classes)
	else:
		print("Model doesn't exist")
		exit()
	
	T = torch.nn.DataParallel(T).cuda()
	#E = facenet.FaceNet(n_classes)
	#E = torch.nn.DataParallel(E).cuda()

	utils.load_my_state_dict(T, ckp_T)
	#utils.load_my_state_dict(E, ckp_E)

	closs = losses.CrossEntropyLoss().cuda()
	
	T.eval()
	#E.eval()

	'''
	F = facenet.IR_50_112((112, 112))
	BACKBONE_RESUME_ROOT = "eval_model/ir50.pth"
	print("Loading Backbone Checkpoint ")
	F.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
	F = torch.nn.DataParallel(F).cuda()
	F.eval()
	'''

	print("Model initialize finish!")
	print("Start inversion attack!")

	for random_seed in range(1):
		tf = time.time()
		torch.manual_seed(random_seed) # cpu
		torch.cuda.manual_seed(random_seed) #gpu
		np.random.seed(random_seed) #numpy

		z = np.random.randn(bs, 3, 64, 64)
		z = torch.from_numpy(z).to(device).float()
		z = nn.Parameter(z).cuda()
		optimizer = torch.optim.SGD([z], lr=lr, momentum=momentum)

		best_ACC = 0.0

		for i in range(iter_times):
			__, out, ___ = T(z)
			
			Iden_Loss = closs(out, one_hot)
			
			optimizer.zero_grad()
			Iden_Loss.backward()
			optimizer.step()
			
			Iden_Loss_val = Iden_Loss.item()

			if (i+1) % 500 == 0:
				fake_img = z.detach()
				__, __, out_iden = E(utils.low2high(fake_img))
				#center = utils.calc_center(feat, iden)
				#knn = utils.calc_knn(feat, iden)
				center = 0.0
				knn = 0.0
				acc = iden.eq(out_iden.float()).sum().item() * 1.0 / bs
				if acc > best_ACC:
					best_ACC = acc
				print("Iteration:{}\tIden Loss:{:.2f}\tKNN:{:.2f}\tCenter:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Iden_Loss_val, knn, center, acc))

		interval = time.time() - tf
		print("Time:{:.2f}\tBest Acc:{:.2f}".format(interval, best_ACC))

		fake_img = z.detach()
		utils.save_tensor_images(fake_img, "emi_celeba.png", nrow=8)

	
		

	

