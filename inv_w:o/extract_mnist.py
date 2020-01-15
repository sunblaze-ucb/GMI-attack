import torch, sys, os, gc, classify, utils
import numpy as np 
import pandas as pd
import torch.nn as nn


os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
device = "cuda"
path = "./feat_mnist"
os.makedirs(path, exist_ok=True)

embed_size = 512
n_classes = 10
tot = 62995

mode = "knn"
pre_path = "eval_model/SCNN10.tar"

def main(data_loader):
	net = classify.SCNN(n_classes)
	net = torch.nn.DataParallel(net).cuda()
	utils.load_my_state_dict(net, torch.load(pre_path)['state_dict'])

	net.eval()
	idx = 0

	if mode == "center":
		with torch.no_grad():
			features = torch.zeros(n_classes, embed_size).cuda()
			num = torch.zeros(n_classes, 1).cuda()
			for img, one_hot, iden in data_loader:
				img, one_hot, iden = img.to(device), one_hot.to(device), iden.to(device)
				feat, __ = net(img)
				bs = img.size(0)
				iden = iden.view(-1)
				for i in range(bs):
					real_iden = iden[i].item()
					features[real_iden, :] += feat[i, :]
					num[real_iden] += 1
					idx += 1

		features /= num
		features = features.detach().cpu().numpy()
		np.save(os.path.join(path, "center.npy"), features)

		print("Finish Center:{}".format(idx))
	
	else:
		with torch.no_grad():
			features = torch.zeros(tot, embed_size).cuda()
			info = torch.zeros(tot, 1).cuda()
			
			for img, one_hot, iden in data_loader:
				img, one_hot, iden = img.to(device), one_hot.to(device), iden.to(device)
				feat, __ = net(img)
				bs = img.size(0)
				iden = iden.view(-1)
				for i in range(bs):
					real_iden = iden[i].item()
					features[idx, :] += feat[i, :]
					info[idx] = real_iden
					idx += 1

		features = features.detach().cpu().numpy()
		info = info.detach().cpu().numpy()
		np.save(os.path.join(path, "feat.npy"), features)
		np.save(os.path.join(path, "info.npy"), info)

		print("Finish KNN:{}".format(idx))

dataset_name = "MNIST"

if __name__ == '__main__':
	
	file = "./config/" + dataset_name + ".json"
	args = utils.load_params(json_file=file)
	file_path = args['dataset']['test_file_path']
	print("Dataset Path:" + file_path)
	
	data_set, data_loader = utils.init_dataloader(args, file_path, mode="attack")
	main(data_loader)
	
	






		