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

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

target = 1

if __name__ == '__main__':
	
	bs = 100
	n_classes = 1000
	one_hot, iden = torch.zeros(bs, n_classes), torch.zeros(bs)
	for i in range(bs):
		one_hot[i, i] = 1
		iden[i] = i
	
	iden = iden.view(-1).float().cuda()
	one_hot = one_hot.float().cuda()
	
	#model_name in ["VGG16", "IR152", "FaceNet64"]

	if target == 1:
		attack_name = "VGG16_set1"
		eval_name = "FaceNet"
	elif target == 2:
		attack_name = "IR152_set1"
		eval_name = "FaceNet"
	elif target == 3:
		attack_name = "FaceNet64"
		eval_name = "FaceNet"
	
	
	print("Attack model name:" + attack_name)
	print("Eval name:" + eval_name)
	print("Iter times:{}".format(iter_times))
	print("lamda:{}".format(lamda))
	print("LR:{:.2f}".format(lr))

	path_G = "result_model/celeba_G_v2.tar"
	path_D = "result_model/celeba_D_v2.tar"
	path_T = "attack_model/" + attack_name + ".tar"
	ckp_G = torch.load(path_G)['state_dict']
	ckp_D = torch.load(path_D)['state_dict']
	ckp_T = torch.load(path_T)['state_dict']
	
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
	F = facenet.IR_50_112((112, 112))
	BACKBONE_RESUME_ROOT = "eval_model/ir50.pth"
	print("Loading Backbone Checkpoint ")
	F.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
	F = torch.nn.DataParallel(F).cuda()
	F.eval()

	utils.load_my_state_dict(G, ckp_G)
	utils.load_my_state_dict(D, ckp_D)
	utils.load_my_state_dict(T, ckp_T)
	
	closs = losses.CrossEntropyLoss().cuda()
	
	D.eval()
	G.eval()
	T.eval()
	
	print("Model initialize finish!")
	print("Start inversion attack!")


	min_center, min_knn = 0, 0

	for random_seed in range(1):
		tf = time.time()
		torch.manual_seed(random_seed) # cpu
		torch.cuda.manual_seed(random_seed) #gpu
		np.random.seed(random_seed) #numpy

		z = np.random.randn(bs, 100)
		z = torch.from_numpy(z).to(device).float()
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
		feat = F(utils.low2high(fake))
		center, knn = utils.calc_center(feat, iden), utils.calc_knn(feat, iden)

		interval = time.time() - tf
		print("Time:{:.2f}\tCenter:{:.2f}\tKnn:{:.2f}".format(interval, center, knn))

	


	#fake_img = G(z.detach())
		#utils.save_tensor_images(fake_img, "inv_celeba_ir152.png", nrow=8)
	
		

	

