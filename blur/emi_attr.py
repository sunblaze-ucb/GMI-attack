import torch, sys, os, time, random, losses
import numpy as np 
import torch.nn as nn
from utils import *
from classify import *
from facenet import *
from discri import DWGANGP
from gen import BlurNet


ld_input_size = 32

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
device = "cuda"
target = 1
eval_id = 2

n_classes = 1000
iter_times = 2
momentum = 0.9
lamda = 100
lr = 2e-2
bs = 100

output_dir = 'result'
result_imgdir = os.path.join(output_dir, "deblur_images")
os.makedirs(result_imgdir, exist_ok=True)

def completion_network_loss(input, output):
    bs = input.size(0)
    loss = torch.sum(torch.abs(output - input)) / bs
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

	
	print("Model Initalize")
	I = FaceAtt()
	BACKBONE_RESUME_ROOT = "eval_model/FaceNet_att.tar"
	ckp_I = torch.load(BACKBONE_RESUME_ROOT)
	I = torch.nn.DataParallel(I).cuda()
	load_my_state_dict(I, ckp_I['state_dict'])
	print("Loading Backbone Checkpoint ")
	I.eval()
	
	root_path = "result_model"

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
	ckp_V = torch.load(path_V)
	load_state_dict(V, ckp_V['state_dict'])

	V.eval()
	

	cnt = 0

	for blur_img, real_img, one_hot, attr in data_loader:
		blur_img, real_img, one_hot, attr = blur_img.to(device), real_img.to(device), one_hot.to(device), attr.to(device)
		
		for random_seed in range(5):

			tf = time.time()
			torch.manual_seed(random_seed) # cpu
			torch.cuda.manual_seed(random_seed) #gpu
			np.random.seed(random_seed) #numpy
			random.seed(random_seed)

			z = torch.randn(bs, 3, 64, 64).to(device).float()
			z.requires_grad = True
			v = torch.zeros(z.size()).to(device).float()

			for i in range(iter_times):
				__, out_prob, out_iden = V(z)

				if z.grad is not None:
					z.grad.data.zero_()

				Prior_Loss = completion_network_loss(z, blur_img)
				Iden_Loss = criteria(out_prob, one_hot) 
				Total_Loss = Prior_Loss + lamda * Iden_Loss
				
				Total_Loss.backward()
				
				v_prev = v.clone()
				gradient = z.grad.data
				v = momentum * v - lr * gradient
				z = z + ( - momentum * v_prev + (1 + momentum) * v)
				z = z.detach()
				z = torch.clamp(z, 0, 1)
				z.requires_grad = True

		out = I(low2high(z))
		out_cls = (out >= 0.5)
		eq = (out_cls.long() == attr.long()) * 1.0
		acc = torch.mean(eq.float(), dim=0)
		out_str = ''
		for i in range(acc.shape[0]):
			out_str += "%.2f " %(acc[i].item())
		
		print(out_str)
		

	
	
