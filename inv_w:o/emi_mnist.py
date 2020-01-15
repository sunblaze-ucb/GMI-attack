import torch, os, time, random, generator, discri, classify, utils, facenet, losses
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls

device = "cuda"
dataset = "MNIST"
n_classes = 1000
iter_times = 1000
momentum = 0.9
lamda = 100
lr = 2e-2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
	bs = 5
	n_classes = 10
	one_hot, iden = torch.zeros(bs, n_classes), torch.zeros(bs)
	for i in range(bs):
		one_hot[i, i+5] = 1
		iden[i] = i+5
	
	iden = iden.view(-1).float().cuda()
	one_hot = one_hot.float().cuda()
	
	#model_name in ["Lenet", "VGG16", "FaceNet64"]
	
	attack_name = "MCNN10"
	eval_name = "SCNN10"
	print("Attack model name:" + attack_name)
	print("Eval name:" + eval_name)
	print("Iter times:{}".format(iter_times))
	print("lamda:{}".format(lamda))
	print("LR:{:.2f}".format(lr))

	path_T = "attack_model/" + attack_name + ".tar"
	path_E = "eval_model/" + eval_name + ".tar"
	ckp_T = torch.load(path_T)['state_dict']
	ckp_E = torch.load(path_E)['state_dict']

	if attack_name.startswith("MCNN"):
		T = classify.MCNN(n_classes)
	else:
		print("Model doesn't exist")
		exit()
	
	T = torch.nn.DataParallel(T).cuda()
	E = classify.SCNN(n_classes)
	E = torch.nn.DataParallel(E).cuda()

	utils.load_my_state_dict(T, ckp_T)
	utils.load_my_state_dict(E, ckp_E)

	closs = losses.CrossEntropyLoss().cuda()
	
	T.eval()
	E.eval()

	print("Model initialize finish!")
	print("Start inversion attack!")

	max_score = torch.zeros(bs)
	max_iden = torch.zeros(bs)
	z_hat = torch.zeros(bs, 1, 32, 32)

	for random_seed in range(5):
		tf = time.time()
		torch.manual_seed(random_seed) # cpu
		torch.cuda.manual_seed(random_seed) #gpu
		np.random.seed(random_seed) #numpy
		random.seed(random_seed)

		z = torch.zeros(bs, 1, 32, 32).to(device).float()
		v = torch.zeros(z.size()).to(device).float()
		z.requires_grad = True
		
		for i in range(iter_times):
			if z.grad is not None:
				z.grad.data.zero_()

			out = T(z)
			Iden_Loss = closs(out, one_hot)
			Total_Loss = lamda * Iden_Loss

			#optimizer.zero_grad()
			Total_Loss.backward()
			#optimizer.step()

			v_prev = v.clone()
			gradient = z.grad.data
			v = momentum * v - lr * gradient
			z = z + ( - momentum * v_prev + (1 + momentum) * v)
			z = torch.clamp(z.detach(), 0, 1).float()
			z.requires_grad = True
			
			Iden_Loss_val = Iden_Loss.item()

			if (i+1) % 500 == 0:
				__, eval_out = E(z.detach())
				out_iden = torch.argmax(eval_out, dim=1)
				acc = iden.eq(out_iden.float()).sum().item() * 1.0 / bs
				
				print("Iteration:{}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Iden_Loss_val, acc))

		fake = z.clone()
		score = T(fake)
		__, eval_out = E(fake)
		out_iden = torch.argmax(eval_out, dim=1)

		cnt = 0

		for i in range(bs):
			if score[i, i+5].item() > max_score[i].item():
				max_score[i] = score[i, i+5]
				max_iden[i] = out_iden[i]
				z_hat[i, :, :, :] = z[i, :, :, :]

			if out_iden[i].item() == i+5:
				cnt += 1
				
		acc = cnt * 1.0 / bs
		interval = time.time() - tf
		print("Time:{:.2f}\tAcc:{:.2f}".format(interval, acc))

	correct = 0
	for i in range(bs):
		if max_iden[i].item() == i+5:
			correct += 1

	feat, __ = E(z_hat)
	acc = correct * 1.0 / bs
	center, knn = utils.calc_center(feat, iden, "feat_mnist"), utils.calc_knn(feat, iden, "feat_mnist")
	print("Acc:{:.2f}\tCenter:{:.2f}\tKNN:{:.2f}".format(acc, center, knn))
	

	utils.save_tensor_images(z_hat, "emi_mnist.png", nrow=8)
	
		

	

