import torch, os, time, random, generator, discri, classify, utils, facenet, losses
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls

device = "cuda"
dataset = "MNIST"
n_classes = 10
iter_times = 3000
momentum = 0.9
lamda = 100
lr = 2e-2


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 2 lbl_list = [4, 5, 6, 7, 9]
# 3 lbl_list = [2, 4, 6, 7, 9]

# 1 lbl_list = [5, 6, 7, 8, 9]

lbl_list = [2, 4, 6, 7, 9]


if __name__ == '__main__':
	bs = 5
	n_classes = 10
	one_hot, iden = torch.zeros(bs, n_classes), torch.zeros(bs)
	for i in range(bs):
		one_hot[i, lbl_list[i]] = 1
		iden[i] = lbl_list[i]
	
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

	path_G = "result_model/MNIST_G3.tar"
	path_D = "result_model/MNIST_D3.tar"
	path_T = "attack_model/" + attack_name + ".tar"
	path_E = "eval_model/" + eval_name + ".tar"
	ckp_G = torch.load(path_G)['state_dict']
	ckp_D = torch.load(path_D)['state_dict']
	ckp_T = torch.load(path_T)['state_dict']
	ckp_E = torch.load(path_E)['state_dict']

	D = discri.DGWGAN32(in_dim=1)
	G = generator.GeneratorMNIST(100)	
	G = torch.nn.DataParallel(G).cuda()
	D = torch.nn.DataParallel(D).cuda()
	
	if attack_name.startswith("MCNN"):
		T = classify.MCNN(n_classes)
	else:
		print("Model doesn't exist")
		exit()
	
	T = torch.nn.DataParallel(T).cuda()
	E = classify.SCNN(n_classes)
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

	max_score = torch.zeros(bs)
	max_iden = torch.zeros(bs)
	z_hat = torch.zeros(bs, 100)

	for random_seed in range(10):
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
			out = T(fake)
			
			if z.grad is not None:
				z.grad.data.zero_()

			Prior_Loss = - label.mean()
			Iden_Loss = closs(out, one_hot)
			Total_Loss = Prior_Loss + lamda * Iden_Loss

			#optimizer.zero_grad()
			Total_Loss.backward()
			#optimizer.step()

			v_prev = v.clone()
			gradient = z.grad.data
			v = momentum * v - lr * gradient
			z = z + ( - momentum * v_prev + (1 + momentum) * v)
			z = torch.clamp(z.detach(), -1, 1).float()
			z.requires_grad = True
			
			Prior_Loss_val = Prior_Loss.item()
			Iden_Loss_val = Iden_Loss.item()

			
			if (i+1) % 500 == 0:
				fake_img = G(z.detach())
				__, eval_out = E(fake_img)
				out_iden = torch.argmax(eval_out, dim=1)
				#center = utils.calc_center(feat, iden)
				#knn = utils.calc_knn(feat, iden)
				#center = 0.0
				#knn = 0.0
				acc = iden.eq(out_iden.float()).sum().item() * 1.0 / bs
				
				print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Prior_Loss_val, Iden_Loss_val, acc))

		fake = G(z)
		score = T(fake)
		__, eval_out = E(fake)
		out_iden = torch.argmax(eval_out, dim=1)

		cnt = 0

		for i in range(bs):
			if score[i, lbl_list[i]].item() > max_score[i].item():
				max_score[i] = score[i, lbl_list[i]]
				max_iden[i] = out_iden[i]
				z_hat[i, :] = z[i, :]

			if out_iden[i].item() == lbl_list[i]:
				cnt += 1
				
		acc = cnt * 1.0 / bs
		interval = time.time() - tf
		print("Time:{:.2f}\tAcc:{:.2f}".format(interval, acc))

		#utils.save_tensor_images(fake, "inv_mnist_{}.png".format(random_seed), nrow=8)

	correct = 0
	for i in range(bs):
		if max_iden[i].item() == lbl_list[i]:
			correct += 1

	z_hat = z_hat.to(device)
	fake_img = G(z_hat)
	feat, __ = E(fake_img)
	acc = correct * 1.0 / bs
	center, knn = utils.calc_center(feat, iden, "feat_mnist"), utils.calc_knn(feat, iden, "feat_mnist")
	print("Acc:{:.2f}\tCenter:{:.2f}\tKNN:{:.2f}".format(acc, center, knn))
	#utils.save_tensor_images(fake_img, "inv_mnist.png", nrow=8)

	
		

	

