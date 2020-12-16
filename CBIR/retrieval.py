import json
from tqdm import tqdm
import pickle
import vptree
import numpy as np
from scipy.spatial import distance


def train_tree(features_list,dist_measure):
	tree = vptree.VPTree(features_list,dist_measure)
	return tree

def MSE(x,y):
	MSE = np.square(np.subtract(x,y)).mean()

	return MSE

def EUC(x,y):
	EUC = np.linalg.norm(x-y)

	return EUC

def COD(x,y):
	COD = distance.correlation(x,y)

	return COD

'''
with open('data.json') as json_file:
    data = json.load(json_file)
'''
'''
with open('new_data.pkl', 'rb') as json_file:
    new_data = pickle.load(json_file)
'''
with open('final_data.pkl', 'rb') as json_file:
    data = pickle.load(json_file)

feat_list = [np.array(key) for key in data]

tree = vptree.VPTree(feat_list, COD)

c = tree.get_n_nearest_neighbors(feat_list[3], 10)

print(data[tuple(feat_list[3])])
#print(c)

for d_p in c:
	print(data[tuple(d_p[1].tolist())])

raise()
'''
new_path_list = [new_data[key] for key in new_data]

old_path_list = [key for key in data]
#duplicate_path_list = [p for p in old_path_list if p not in old_path_list]
duplicate_path_list = list(set(old_path_list) - set(new_path_list))

duplicate_data = {}
duplicate_key = []

for path in tqdm(duplicate_path_list):

	feat = tuple(data[path])
	if feat not in duplicate_key:
		duplicate_key.append(feat)

		duplicate_data[feat] = [path]
	else:
		duplicate_data[feat].append(path)

new_data.update(duplicate_data)

def write_json(data, filename): 
    with open(filename,'wb') as f: 
        pickle.dump(data, f) 
#print(new_data)
write_json(new_data,'final_data.pkl')
raise()



feature_list = [data[key] for key in data]

#print(len(feature_list))

index = 0

dupllicate_list = []

new_data = {}

for path in tqdm(data):

	feat = data[path]

	ft_list = [] + feature_list

	ft_list.pop(index)

	if feat not in ft_list:

		new_data[tuple(feat)] = path

	index+=1

print(len(new_data))
print(len(data))
'''
'''
for path in tqdm(range(len(data))):

	ft_list = [] + feature_list

	ft_list.pop(index)
	
	if feature_list[index] in ft_list:
		if feature_list[index] not in dupllicate_list:
			dupllicate_list.append(ft_list)

	index+=1
'''

def write_json(data, filename): 
    with open(filename,'wb') as f: 
        pickle.dump(data, f) 
#print(new_data)
write_json(new_data,'new_data.pkl')
'''

with open('data.json') as json_file:
    data = json.load(json_file)


data_dict_ = {data[key]:key for key in data if data[key] not in dupllicate_list}

def write_json(data, filename): 
    with open(filename,'w') as f: 
        json.dump(data, f) 

write_json(data_dict_,'tree_data.json')
'''