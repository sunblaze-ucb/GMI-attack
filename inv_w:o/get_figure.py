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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

target = 3

img_path = "./inv_img"
os.makedirs(img_path, exist_ok=True)

if __name__ == '__main__':
	
	bs = 100
	n_classes = 1000
	one_hot, iden = torch.zeros(bs, n_classes), torch.zeros(bs)
	for i in range(bs):
		one_hot[i, i] = 1
		iden[i] = i
	

	iden = iden.view(-1, 1).float().cuda()
	one_hot = one_hot.float().cuda()
	
	#model_name in ["VGG16", "IR152", "FaceNet64"]

	if target == 1:
		attack_name = "VGG16"
		eval_name = "FaceNet_set2"
	elif target == 2:
		attack_name = "IR152"
		eval_name = "FaceNet_set2"
	elif target == 3:
		attack_name = "FaceNet64"
		eval_name = "FaceNet_set2"
	
	
	print("Attack model name:" + attack_name)
	print("Eval name:" + eval_name)
	print("Iter times:{}".format(iter_times))
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
	
	if attack_name.startswith("Lenet"):
		T = classify.Lenet(n_classes)
	elif attack_name.startswith("VGG16"):
		T = classify.VGG16(n_classes)
	elif attack_name.startswith("IR50"):
		T = classify.IR50(n_classes)
	elif attack_name.startswith("IR152"):
		T = classify.IR152(n_classes)
	elif attack_name.startswith("FaceNet64"):
		T = facenet.FaceNet64(n_classes)
	else:
		print("Model doesn't exist")
		exit()
	
	T = torch.nn.DataParallel(T).cuda()
	E = facenet.FaceNet(n_classes)
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

	z_list, iden_list = [], []
	max_cnt, max_cnt_5 = 0, 0

	flag = torch.zeros(100)
	max_score = torch.zeros(100)
	max_iden = torch.zeros(100)

	z_hat = torch.zeros(bs, 100).to(device).float()


	for random_seed in range(5):
		tf = time.time()
		
		torch.manual_seed(random_seed) # cpu
		torch.cuda.manual_seed(random_seed) #gpu
		np.random.seed(random_seed) #numpy
		random.seed(random_seed)

		z = torch.randn(bs, 100).to(device).float()
		z.requires_grad = True
		#z = nn.Parameter(z).cuda()
		#optimizer = torch.optim.SGD([z], lr=lr, momentum=momentum)

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

			#optimizer.zero_grad()
			Total_Loss.backward()
			#optimizer.step()
			v_prev = v.clone()
			gradient = z.grad.data
			v = momentum * v - lr * gradient
			z = z + ( - momentum * v_prev + (1 + momentum) * v)
			z = torch.clamp(z.detach(), -1, 1).float()
			z.requires_grad = True

		fake = G(z)
		__, out_prob, out_iden = E(utils.low2high(fake))
		out_prob = out_prob.detach().cpu()
		out_iden = out_iden.detach().cpu()
		__, score, __ = T(fake)
		score = score.detach().cpu()
		
		out_iden = out_iden.view(-1)
		cnt = 0

		for i in range(bs):
			if score[i, i].item() > max_score[i].item():
				max_score[i] = score[i, i]
				max_iden[i] = out_iden[i]
			if out_iden[i].item() == i:
				cnt += 1
				flag[i] = 1
				z_hat[i, :] = z[i, :]
				
		interval = time.time() - tf
		print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0/ 100))

	correct = 0
	for i in range(bs):
		if max_iden[i].item() == i:
			correct += 1
	
	correct_5 = torch.sum(flag)
	acc, acc_5 = correct * 1.0 / bs, correct_5 * 1.0 / bs
	print("Acc:{:.2f}\tAcc5:{:.2f}".format(acc, acc_5))

	fake_img = G(z_hat)
	utils.save_tensor_images(fake_img, os.path.join(img_path, "inv_celeba.png"), nrow=8)
	
		

	

