import torch, sys, os, time, random, losses
import numpy as np 
import torch.nn as nn
import utils
from discri import DLWGAN, DGWGAN
from generator import InversionNet

device = "cuda"
output_dir = "./img_iden"
os.makedirs(output_dir, exist_ok=True)
bs = 64

if __name__ == '__main__':
	dataset_name = "celeba"
	file = "./config/" + dataset_name + ".json"
	args = utils.load_params(json_file=file)
	file_path = args['dataset']['test_file_path']

	data_set, data_loader = utils.init_dataloader(args, file_path, bs, mode="attack")
	iden_list = []
	for i in range(100):
		iden_list.append([])

	for img, one_hot, iden in data_loader:
		bs = img.size(0)
		for i in range(bs):
			real_iden = iden[i].item()
			if real_iden >= 100:
				continue
			iden_list[real_iden].append(img[i, :, :, :].unsqueeze(0))

	for i in range(100):
		iden_img = torch.cat(iden_list[i], dim=0)
		print(iden_img.size())
		utils.save_tensor_images(iden_img, os.path.join(output_dir, "inv_iden_{}.png".format(i)), nrow=8)
	
	
