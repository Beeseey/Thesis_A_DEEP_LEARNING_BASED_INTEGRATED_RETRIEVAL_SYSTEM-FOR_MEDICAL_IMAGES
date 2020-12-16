from auto_enc import auto_encoder, auto_encoder_image
from glob import glob
import os
import random
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
from PIL import ImageFile,Image
import numpy as np
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf
import tensorflow_addons as tfa
import keras.backend.tensorflow_backend as tfb

POS_WEIGHT = 10  # multiplier for positive targets, needs to be tuned

random.seed(3)

from keras import backend as K

paths = 'images/*'

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
	image = preprocess_input(image)
	#image = np.expand_dims(image,axis=0)
	#print(image.shape)
	#image = np.resize(image,(224,224,3))
	#image = image.astype("float32")/255.0
	return image

def generator(paths, batch_Size):

	X1, y = list(), list()
	count = 0
	random.shuffle(paths)

	while True:
		for path in paths:
			
			X = load_image(path,(224,224))				
			X1.append(X)
			y.append(X)
			count+=1
			
			if count == batch_Size:
				count = 0
				yield np.array(X1), np.array(y)
				X1, y = list(), list()

path_list = [path for path in glob(paths)]

data_length = len(path_list)

train_data = path_list[:int(data_length*0.75)]
#train_data = dataset[2500:11000]
val_data = path_list[int(data_length*0.75):]

train_gen = generator(train_data,10)

val_gen = generator(val_data,10)

filepath = 'auto_enc_latest.h5'

model = auto_encoder(224,224,3)
#model = auto_encoder_image()

print(model.summary())
#raise()

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.fit(train_gen, epochs=1000,
			verbose=1, steps_per_epoch=len(train_data)//10 ,
			validation_data = val_gen, 
			validation_steps=len(val_data)//10, callbacks=[checkpoint])