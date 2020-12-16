import os
from glob import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
from PIL import ImageFile,Image
import json
from tqdm import tqdm
from functools import partial
from scipy.spatial import distance
import tensorflow as tf
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

images_path = 'images/*'

model_name = 'dn_4_ep1000_lr0.001_mse_0.89.h5'
model = load_model(model_name)

dataset_groups = [os.path.split(group)[-1] for group in glob('../../Patch_thesis/test/*')]

#a = json.load('data.json')


def load_image(path, target_size):
	
	try:
		image = load_img(path, target_size=target_size)
		# convert the image pixels to a numpy array
		image = img_to_array(image)
	except:
		print(path)
		image = cv2.imread(path)
		image =  cv2.resize(image, target_size, interpolation = cv2.INTER_AREA)
		image = Image.fromarray(image)
		image = img_to_array(image)
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#image = np.expand_dims(image,axis=-1)
	image = preprocess_input(image)
	#image = np.expand_dims(image,axis=0)
	
	#image = image.astype("float32")/255.0
	return image

w,h = 32,32
count = 0
x = [0, 0, 0, 0, 0, 0, 0, 32, 32, 32, 32, 32, 32, 32, 
		64, 64, 64, 64, 64, 64, 64, 96, 96, 96, 96, 96, 96, 96, 
		128, 128, 128, 128, 128, 128, 128, 160, 160, 160, 160, 160, 160, 160, 
		192, 192, 192, 192, 192, 192, 192]
y = [0, 32, 64, 96, 128, 160, 192, 0, 32, 64, 96, 128, 160, 192,
	 0, 32, 64, 96, 128, 160, 192, 0, 32, 64, 96, 128, 160, 192,
	  0, 32, 64, 96, 128, 160, 192, 0, 32, 64, 96, 128, 160, 192,
	   0, 32, 64, 96, 128, 160, 192]

paths = [path for path in glob(images_path)]

labels = []

data_dict = dict.fromkeys(paths)

for i in tqdm(range(len(paths))):

	path = paths[i]

	X = load_image(path,(224,224))

	

	img = [X[y[i]:y[i]+h, x[i]:x[i]+w] for i in range(49)]

	label = np.argmax(model.predict(np.array(img)),axis=-1).tolist()

	data_dict[path] = label
	#print(label)
	#labels.append(label)


def write_json(data, filename='data_.json'): 
    with open(filename,'w') as f: 
        json.dump(data, f) 

write_json(data_dict)
