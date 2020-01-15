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

if __name__ == '__main__':
	
	n_classes = 1000
	one_hot, iden = torch.zeros(bs, n_classes), torch.zeros(bs)
	for i in range(bs):
		one_hot[i, i] = 1
		iden[i] = i
	
	iden = iden.view(-1, 1).float().cuda()
	one_hot = one_hot.float().cuda()
	
	# model_name in ["VGG16", "IR152", "FaceNet64"]

	attack_name = "VGG16_set2"
	eval_name = "FaceNet_set2"
	
	
	print("Attack model name:" + attack_name)
	print("Attack number:{}".format(bs))
	print("Eval name:" + eval_name)
	print("Iter times:{}".format(iter_times))
	print("Clip range:{}".format(clip_range))
	print("lamda:{}".format(lamda))
	print("LR:{:.2f}".format(lr))

	path_G = "result_model/celeba_G.tar"
	path_D = "result_model/celeba_D.tar"
	#path_G = "result/models_celeba_gan/celeba_G.tar"
	#path_D = "result/models_celeba_gan/celeba_D.tar"
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
	elif attack_name.startswith('IR50_vib'):
		T = classify.IR50_vib(n_classes)
	elif attack_name.startswith("IR50"):
		T = classify.IR50(n_classes)
	elif attack_name.startswith("IR152_vib"):
		T = classify.IR152_vib(n_classes)
	elif attack_name.startswith("IR152"):
		T = classify.IR152(n_classes)
	elif attack_name.startswith("FaceNet64"):
		T = facenet.FaceNet64(n_classes)
	else:
		print("Model doesn't exist")
		exit()
	
	T = torch.nn.DataParallel(T).cuda()
	#E = facenet.FaceNet(n_classes)
	#E = torch.nn.DataParallel(E).cuda()

	utils.load_my_state_dict(G, ckp_G)
	utils.load_my_state_dict(D, ckp_D)
	utils.load_my_state_dict(T, ckp_T)
	#utils.load_my_state_dict(E, ckp_E)
	
	closs = losses.CrossEntropyLoss().cuda()
	
	D.eval()
	G.eval()
	T.eval()
	#E.eval()
	F = facenet.IR_50_112((112, 112))
	BACKBONE_RESUME_ROOT = "eval_model/ir50.pth"
	print("Loading Backbone Checkpoint ")
	F.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
	F = torch.nn.DataParallel(F).cuda()
	F.eval()

	print("Model initialize finish!")
	print("Start inversion attack!")

	z_list, iden_list = [], []
	max_cnt, max_cnt_5 = 0, 0

	flag = torch.zeros(bs)
	max_score = torch.zeros(bs)
	max_iden = torch.zeros(bs)


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
				fake_img = G(z.detach())
				__, __, out_iden = E(utils.low2high(fake_img))
				acc = iden.eq(out_iden.float()).sum().item() * 1.0 / bs
				if acc > best_ACC:
					best_ACC = acc
				print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Prior_Loss_val, Iden_Loss_val, acc))
			
		fake = G(z)

		feat = F(utils.low2high(fake))

		#__, out_prob, out_iden = E(utils.low2high(fake))
		#out_prob = out_prob.detach().cpu()
		#out_iden = out_iden.detach().cpu()
		#__, score, __ = T(fake)
		#score = score.detach().cpu()
		'''
		out_iden = out_iden.view(-1)
		cnt = 0

		for i in range(bs):
			if score[i, i].item() > max_score[i].item():
				max_score[i] = score[i, i]
				max_iden[i] = out_iden[i]
			if out_iden[i].item() == i:
				cnt += 1
				flag[i] = 1
		'''

		k_dist, center_dist = utils.calc_center(feat, iden), utils.calc_knn(feat, iden)
		print("KNN:{:.2f}\tCenter:{:.2f}".format(k_dist, center_dist))
				
		interval = time.time() - tf
		#print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / 100))

	'''
	correct = 0
	for i in range(bs):
		if max_iden[i].item() == i:
			correct += 1
	
	correct_5 = torch.sum(flag)
	acc, acc_5 = correct * 1.0 / bs, correct_5 * 1.0 / bs
	print("Acc:{:.2f}\tAcc5:{:.2f}".format(acc, acc_5))
	'''

	'''
	fake_img = G(z.detach())
	utils.save_tensor_images(fake_img, "inv_celeba_pii.png", nrow=8)
	'''
	
		

	

