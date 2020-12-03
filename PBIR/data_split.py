from glob import glob
import os
import random
import numpy as np

random.seed(45)

DATASET = 'Patch_SVM_32_60'
PATH_DICT = {}
PATH_DICT_ = {}


dataset_groups = [os.path.split(group)[-1] for group in glob(DATASET+'/*')]
for grp in dataset_groups:
	PATH_DICT[grp] = []
#path_list = [path for path in ]
#print(PATH_DICT)
for path in glob(DATASET+'/*/*'):
	for grp  in dataset_groups:
		if grp in path:
			PATH_DICT[grp].append(path)
		

for div in ["training", "test", "vallidation"]:
	PATH_DICT_[div] = {}
	for grp  in dataset_groups:
		grp_lst = PATH_DICT[grp]
		random.shuffle(grp_lst)

		train_no = int(len(grp_lst)*0.7)
		test_no = int(len(grp_lst)*0.9)
		#val_no = int(len(grp_lst)*0.1)
		if div == "training":
			PATH_DICT_[div][grp] =  grp_lst[:train_no]
		elif div == "test":
			PATH_DICT_[div][grp] =  grp_lst[train_no:test_no]
		elif div == "vallidation":
			PATH_DICT_[div][grp] =  grp_lst[test_no:]


os.system("mkdir training test vallidation")


for div in ["training", "test", "vallidation"]:
	for grp in PATH_DICT:
		p = os.path.join(div,grp)
		os.system("mkdir "+p)

for split in PATH_DICT_:
	for grp in PATH_DICT_[split]:
		p = os.path.join(split,grp)
		for img in PATH_DICT_[split][grp]:
			os.system('cp '+img+' '+p)
