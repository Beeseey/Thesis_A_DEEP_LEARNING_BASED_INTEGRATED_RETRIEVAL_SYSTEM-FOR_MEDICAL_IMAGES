import os
from glob import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import preprocess_input
import cv2
from PIL import ImageFile,Image
import numpy as np
from models import v_res, sae, dn, sae2
import random
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

ImageFile.LOAD_TRUNCATED_IMAGES = True 


DATASET = 'test'
random.seed(45)

dataset_groups = [os.path.split(group)[-1] for group in glob(DATASET+'/*')]


for path in glob('training'+'/*/*'):
	if 'Thumbs' in path:
		os.system('rm '+ path)
for path in glob('vallidation'+'/*/*'):
	if 'Thumbs' in path:
		os.system('rm '+ path)

def OHE(index,length):

	label = np.zeros(length)

	label[index] = 1

	return label

def generator(data, batch_Size, label_len):

	X1, y = list(), list()
	count = 0
	random.shuffle(data)

	while True:
		for path in data:

			X = load_image(path,(32,32))
			label = os.path.split(path)[-2].split('\\')[-1]
			y_ = OHE(dataset_groups.index(label),label_len)

			X1.append(X)
			y.append(y_)
			count+=1
			
			if count == batch_Size:
				yield np.array(X1), np.array(y)
				X1, y = list(), list()
				count = 0

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
	#image = np.expand_dims(image,axis=-1)
	#image = image.astype("float32")/255.0
	return image

train_data = glob('training'+'/*/*')
val_data = glob('vallidation'+'/*/*')

train_gen = generator(train_data,10,len(dataset_groups))
val_gen = generator(val_data,10,len(dataset_groups))


EPOCHS = 1000
INIT_LR = 1e-3

OPT = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

#model = v_res(model_name='RESNET50')

filepath = 'dn_5_ep1000_lr0.001_mse.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model = dn(32,32,3)
model.compile(optimizer= OPT, loss='mse', metrics=['accuracy'])

model.fit(train_gen, epochs=1000,
			verbose=1, steps_per_epoch=len(train_data)//10 ,
			validation_data = val_gen, 
			validation_steps=len(val_data)//10, callbacks=[checkpoint])


#print(next(g))


'''
for path in glob(DATASET+'/*/*'):
	label = os.path.split(path)[-2].split('\\')[-1]
	print(dataset_groups.index(label))

'''