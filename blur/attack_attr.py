import torch, sys, os, time, random, losses
import numpy as np 
import torch.nn as nn
from utils import *
from classify import *
from facenet import *
from discri import DWGANGP
from gen import BlurNet


ld_input_size = 32

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
device = "cuda"
target = 1

n_classes = 1000
iter_times = 3000
momentum = 0.9
lamda = 100
lr = 2e-2
bs = 100

output_dir = 'result'
result_imgdir = os.path.join(output_dir, "deblur_images")
os.makedirs(result_imgdir, exist_ok=True)

def noise_loss(V, img1, img2):
    feat1, __, ___ = V(img1)
    feat2, __, ___ = V(img2)
    
    loss = torch.mean(torch.abs(feat1 - feat2))
    #return mse_loss(output * mask, input * mask)
    return loss

def load_state_dict(self, state_dict):
	own_state = self.state_dict()
	
	for name, param in state_dict.items():
		if name == 'module.fc_layer.0.weight':
			own_state['module.fc_layer.weight'].copy_(param.data)
		elif name == 'module.fc_layer.0.bias':
			own_state['module.fc_layer.bias'].copy_(param.data)
		elif name in own_state:
			own_state[name].copy_(param.data)
		else:
			print(name)

def test_net(net, test_loader):
	net.eval()
	ACC_cnt = AverageMeter()
	for i, (blur, img, one_hot, iden) in enumerate(test_loader):
		img, one_hot, iden = img.to(device), one_hot.to(device), iden.to(device)
		bs = img.size(0)
		iden = iden.view(-1, 1)
		
		___, out_prob, out_iden = net(img)
		out_iden = out_iden.view(-1, 1)
		ACC = torch.sum(out_iden == iden).item() / bs
		ACC_cnt.update(ACC, bs)
		
	print("Test ACC:{:.2f}".format(ACC_cnt.avg * 100))

if __name__ == '__main__':

	dataset_name = "celeba"
	file = "./config/" + dataset_name + ".json"
	args = load_params(json_file=file)

	if target == 1:
		attack_name = "VGG16"
	elif target == 2:
		attack_name = "IR152"
	else:
		attack_name = "FaceNet64"
		


	print("Attack Model Name:" + attack_name)
	print("Iter times:{}".format(iter_times))
	print("Lamda:{:.2f}".format(lamda))
	print("LR:{:.3f}".format(lr))

	criteria = losses.CrossEntropyLoss().cuda()
	file_path = args['dataset']['test_file_path']
	
	if dataset_name == "celeba":
		data_set, data_loader = init_dataloader(args, file_path, bs, mode="attr")
	
	#model_name in ["VGG16", "IR152", "FaceNet64"]
	#                m1       m2            m3

	'''
	F = facenet.IR_50_112((112, 112))
	BACKBONE_RESUME_ROOT = "eval_model/ir50.pth"
	print("Loading Backbone Checkpoint ")
	F.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
	F = torch.nn.DataParallel(F).cuda()
	F.eval()
	'''
	root_path = "result_model"

	Net = BlurNet()
	Net = torch.nn.DataParallel(Net).cuda()

	D = DWGANGP()
	D = torch.nn.DataParallel(D).cuda()

	if attack_name.startswith("VGG16"):
		V = VGG16(1000)
	elif attack_name.startswith("IR152"):
		V = IR152(1000)
	elif attack_name.startswith("FaceNet64"):
	 	V = FaceNet64(1000)
	else:
		print("Model name Error")
		exit()

	V = torch.nn.DataParallel(V).cuda()
	
	path_V = "attack_model/" + attack_name + ".tar"
	path_N = os.path.join(root_path, "Blur_m2_v5.tar")
	path_D = os.path.join(root_path, "D_m2_v5.tar")
	
	ckp_N = torch.load(path_N)
	ckp_D = torch.load(path_D)
	ckp_V = torch.load(path_V)
	
	load_my_state_dict(Net, ckp_N['state_dict'])
	load_my_state_dict(D, ckp_D['state_dict'])
	load_state_dict(V, ckp_V['state_dict'])

	print("Model Initalize")
	I = FaceAtt()
	BACKBONE_RESUME_ROOT = "eval_model/FaceNet_att.tar"
	ckp_I = torch.load(BACKBONE_RESUME_ROOT)
	I = torch.nn.DataParallel(I).cuda()
	load_my_state_dict(I, ckp_I['state_dict'])
	print("Loading Backbone Checkpoint ")
	
	D.eval()
	Net.eval()
	V.eval()
	I.eval()

	
	best_acc = 0.0



	for blur_img, real_img, one_hot, attr in data_loader:
		
		blur_img, real_img, one_hot, attr = blur_img.to(device), real_img.to(device), one_hot.to(device), attr.to(device)
		
		max_score = torch.zeros(bs)
		flag = torch.zeros(bs)

		for random_seed in range(5):
			tf = time.time()
			torch.manual_seed(random_seed) # cpu
			torch.cuda.manual_seed(random_seed) #gpu
			np.random.seed(random_seed) #numpy
			random.seed(random_seed)

			z = torch.randn(bs, 100).to(device).float()
			z.requires_grad = True
			v = torch.zeros(bs, 100).to(device).float()

			for i in range(iter_times):
				
				output = Net((blur_img, z))
				logit_d = D(output)
				
				__, out_prob, out_iden = V(output)

				if z.grad is not None:
					z.grad.data.zero_()

				Prior_Loss = - logit_d.mean()
				Iden_Loss = criteria(out_prob, one_hot) 
				Total_Loss = lamda * Iden_Loss
				
				Total_Loss.backward()
				
				v_prev = v.clone()
				gradient = z.grad.data
				v = momentum * v - lr * gradient
				z = z + ( - momentum * v_prev + (1 + momentum) * v)
				z = z.detach()
				z = torch.clamp(z, -1, 1)
				z.requires_grad = True

				Prior_Loss_val = Prior_Loss.item()
				Iden_Loss_val = Iden_Loss.item()

				if (i + 1) % 500 == 0:
					print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}".format(i+1, Prior_Loss_val, Iden_Loss_val))
				
			output = Net((blur_img, z))
			out = I(low2high(output))
			out_cls = (out >= 0.5)

			eq = (out_cls.long() == attr.long()) * 1.0
			acc = torch.mean(eq.float(), dim=0)
			out_str = ''
			for i in range(acc.shape[0]):
				out_str += "%.2f " %(acc[i].item())
			
			print(out_str)
			interval = time.time() - tf
			tf = time.time()
			
			
			#print("Time:%.2f\tACC:%.2f\t" % (interval, acc))

		

	

	
	
