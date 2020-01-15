import torch, sys, os, time, random, losses, facenet
import numpy as np 
import torch.nn as nn
from utils import *
from discri import DLWGAN, DGWGAN
from generator import InversionNet

ld_input_size = 32
z_dim = 100

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
device = "cuda"
result_img_dir = "./attack_imgs"
os.makedirs(result_img_dir, exist_ok=True)

def load_state_dict(self, state_dict):
	own_state = self.state_dict()
	
	for name, param in state_dict.items():
		if name == 'module.fc_layer.weight':
			own_state['module.fc_layer.0.weight'].copy_(param.data)
		elif name == 'module.fc_layer.bias':
			own_state['module.fc_layer.0.bias'].copy_(param.data)
		elif name in own_state:
			own_state[name].copy_(param.data)
		else:
			print(name)

if __name__ == '__main__':
	
	dataset_name = "celeba"
	file = "./config/" + dataset_name + ".json"
	args = load_params(json_file = file)
	model_name = args["dataset"]["model_name"]
	device = "cuda"
	output_dir = 'result'
	inpainted_imagedir = os.path.join(output_dir, "inpainted_images")

	iter_times = args['inpainting']['iterations']
	momentum = args['inpainting']['momentum']
	lr = args['inpainting']['learning_rate']
	bs = args['inpainting']['batch_size']
	file_path = args['dataset']['test_file_path']
	img_size = args['dataset']['img_size']
	
	os.makedirs(inpainted_imagedir, exist_ok=True)

	print("---------------------Training [%s]-----------------------" % model_name)
	print_params(args["dataset"], args[model_name], dataset = args['dataset']['name'])
	
	if args['inpainting']['masktype'] == "center":
		mode = 1
	else:
		mode = 2
	mask = get_mask(img_size, bs, mode)
	criteria = losses.CrossEntropyLoss().cuda()
	

	if dataset_name == "celeba":
		data_set, data_loader = init_dataloader(args, file_path, bs, mode="attr")
	else:
		data_set, data_loader = init_pubfig(args, file_path, bs)

	#model_name in ["VGG16", "Lenet", "FaceNet64"]
	#                m1       m2            m3
	
	model_name = "VGG16"
	
	net_path = "./result/mask_model_reg"
	
	path_N = os.path.join(net_path, "Inversion_m1.tar")
	path_DG = os.path.join(net_path, "DG_m1.tar")
	path_DL = os.path.join(net_path, "DL_m1.tar")
	path_V = os.path.join("attack_model", model_name+".tar")
	
	ckp_N = torch.load(path_N)
	ckp_DG = torch.load(path_DG)
	ckp_DL = torch.load(path_DL)
	ckp_V = torch.load(path_V)


	Net = InversionNet()
	Net = torch.nn.DataParallel(Net).cuda()

	DL = DLWGAN()
	DG = DGWGAN()

	DL = torch.nn.DataParallel(DL).cuda()
	DG = torch.nn.DataParallel(DG).cuda()

	V = get_model(model_name, args["dataset"]["num_of_classes"])

	V = torch.nn.DataParallel(V).cuda()

	load_my_state_dict(Net, ckp_N['state_dict'])
	load_my_state_dict(DL, ckp_DL['state_dict'])
	load_my_state_dict(DG, ckp_DG['state_dict'])
	load_state_dict(V, ckp_V['state_dict'])

	
	print("Model Initalize")
	
	I = facenet.FaceAtt()
	BACKBONE_RESUME_ROOT = "eval_model/FaceNet_att.tar"
	ckp_I = torch.load(BACKBONE_RESUME_ROOT)
	I = torch.nn.DataParallel(I).cuda()
	load_my_state_dict(I, ckp_I['state_dict'])
	print("Loading Backbone Checkpoint ")
	
	mpv = np.array([0.5061, 0.4254, 0.3828])
	mpv = torch.tensor(mpv.astype(np.float32).reshape(1, 3, 1, 1)).cuda()

	DG.eval()
	DL.eval()
	Net.eval()
	V.eval()
	I.eval()

	cnt = 0

	for real_img, one_hot, attr in data_loader:
		max_score = torch.zeros(bs)
		max_iden = torch.zeros(bs)
		real_img, one_hot, attr = real_img.to(device), one_hot.to(device), attr.to(device)
		x_mask = real_img - real_img * mask + mpv * mask
		inp = torch.cat((x_mask, mask), dim=1)
		tf = time.time()
		
		for random_seed in range(5):
			torch.manual_seed(random_seed) # cpu
			torch.cuda.manual_seed(random_seed) #gpu
			np.random.seed(random_seed) #numpy
			random.seed(random_seed)
			cnt = 0

			z = np.random.randn(bs, z_dim)
			z = torch.from_numpy(z).to(device).float()
			z.requires_grad = True
			v = torch.zeros(bs, 100).to(device).float()

			for t in range(iter_times):
				
				output = Net((inp, z))
				output = real_img - real_img * mask + output * mask

				hole_area = gen_hole_area((ld_input_size, ld_input_size), (img_size, img_size))
				fake_crop = crop(output, hole_area)
				logit_dl = DL(fake_crop)
				logit_dg = DG(output)
				
				__, out_prob, out_iden = V(output)

				if z.grad is not None:
					z.grad.data.zero_()

				Prior_Loss = - logit_dl.mean() - logit_dg.mean()
				Iden_Loss = criteria(out_prob, one_hot) 
				Total_Loss = 100 * Iden_Loss + Prior_Loss
				
				Total_Loss.backward()
				
				v_prev = v.clone()
				gradient = z.grad.data
				v = momentum * v - lr * gradient
				z = z + ( - momentum * v_prev + (1 + momentum) * v)
				z = torch.clamp(z.detach(), -1, 1)
				z.requires_grad = True

				Prior_Loss_val = Prior_Loss.item()
				Iden_Loss_val = Iden_Loss.item()

				if (t + 1) % 500 == 0:
					print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}".format(t + 1, Prior_Loss_val, Iden_Loss_val))
			
			output = Net((inp, z))
			output = real_img - real_img * mask + output * mask
			res = low2high(output)
			out = I(res)
			
			out_cls = (out >= 0.5)
			eq = (out_cls.long() == attr.long()) * 1.0
			acc = torch.mean(eq.float(), dim=0)
			out_str = ''
			for i in range(acc.shape[0]):
				out_str += "%.2f " %(acc[i].item())
					
			interval = time.time() - tf
			tf = time.time()
			print("Time:%.2f" %(interval))
			print(out_str)
			
		cnt += 1
		if cnt >= 1:
			break

	
	
