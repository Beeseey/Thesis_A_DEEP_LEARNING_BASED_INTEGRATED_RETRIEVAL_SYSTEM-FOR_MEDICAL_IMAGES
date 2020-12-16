import json
from tqdm import tqdm
import pickle
import numpy as np


def count_label(labels):
	
	new_labels = np.zeros(60).tolist()

	for label_index in labels:
		new_labels[label_index]+=1
	return new_labels

with open('caption_features.pkl', 'rb') as json_file:
    data = pickle.load(json_file)


feature_list = [sorted(data[key]) for key in data]

print(feature_list[0])

raise()

#feature_list = [count_label(data[key]) for key in data]

#print(len(feature_list))

index = 0

new_data = {}

for path in tqdm(data):

	feat = data[path]

	feat = sorted(feat)

	ft_list = [] + feature_list

	ft_list.pop(index)

	if feat not in ft_list:

		new_data[tuple(feat)] = path

	index+=1

print('Adding duplicated data')

new_path_list = [new_data[key] for key in new_data]

old_path_list = [key for key in data]

duplicate_path_list = list(set(old_path_list) - set(new_path_list))

duplicate_data = {}
duplicate_key = []

for path in tqdm(duplicate_path_list):

	feat = tuple(sorted(data[path]))
	
	if feat not in duplicate_key:
		duplicate_key.append(feat)

		duplicate_data[feat] = [path]
	else:
		duplicate_data[feat].append(path)

new_data.update(duplicate_data)

def write_json(data, filename): 
    with open(filename,'wb') as f: 
        pickle.dump(data, f) 

write_json(new_data,'restructured_data_captions.pkl')

