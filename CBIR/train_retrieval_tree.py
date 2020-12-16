import json
from tqdm import tqdm
import pickle, json
import vptree
import numpy as np
from scipy.spatial import distance
from sklearn import metrics
from sklearn.metrics import jaccard_similarity_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
sys.setrecursionlimit(10000)

def train_tree(features_list,dist_measure):
	tree = vptree.VPTree(features_list,dist_measure)
	return tree

def MSE(x,y):
	MSE = np.square(np.subtract(x,y)).mean()

	return MSE

def EUC(x,y):
	EUC = np.linalg.norm(x-y)

	return EUC

def JAC(x,y):
	JAC = jaccard_similarity_score(x,y)

	return JAC
'''

def JAC(x,y):
	x = np.expand_dims(x,axis=0)
	y = np.expand_dims(y,axis=0)
	JAC = metrics.pairwise.cosine_similarity(x,y)

	return JAC

print(JAC(np.array([0,0,0,0,0,1]),np.array([0,0,0,0,16,1])))
'''
#raise()
def distancesum (arr, n): 
      
    # for each point, finding  
    # the distance. 
    res = 0
    sum = 0
    for i in range(n): 
        res += (arr[i] * i - sum) 
        sum += arr[i] 
      
    return res 
      
def CTB_2( x , y):
	n = len(x) 
	CTB = distancesum(x, n) + distancesum(y, n)

	return CTB

def CTB(x,y):
	manhattan_distance = distance.cityblock(x, y)/(len(y))

	return manhattan_distance


'''
with open('restructured_data_captions.pkl', 'rb') as json_file:
    data = pickle.load(json_file)
'''
#print(data.keys())

#raise()


res = {}
with open('auto_enc_features_2.pkl', 'rb') as json_file:
    data = pickle.load(json_file)

'''
for key in data:
	print(data[key])
	print(pad_sequences(key,7))
	raise()
'''

#feat_list = [np.array(pad_sequences([sorted(key)],7)) for key in data]

#print(feat_list)

#raise()
feat_list = [np.array(key) for key in data]
#print(len(feat_list))

tree = vptree.VPTree(feat_list, EUC)

#c = tree.get_n_nearest_neighbors(feat_list[10], 1000)

#print(data[tuple(feat_list[10])])
#print(c)

#for d_p in c:
#	print(data[tuple(d_p[1].tolist())])

def write_json(data, filename): 
    with open(filename,'wb') as f: 
        pickle.dump(data, f) 
#print(new_data)
write_json(tree,'trees/EUC_cbir_tree_2.pkl')