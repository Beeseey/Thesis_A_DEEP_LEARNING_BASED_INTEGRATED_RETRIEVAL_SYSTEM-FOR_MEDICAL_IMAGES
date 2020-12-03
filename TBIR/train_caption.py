from keras.preprocessing.text import Tokenizer
import json
from glob import glob
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
from PIL import ImageFile,Image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D, AveragePooling2D, Conv1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout, Flatten, RepeatVector, TimeDistributed, Bidirectional, concatenate, Lambda, dot, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.applications import DenseNet121

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

    image = preprocess_input(image)
    return image

def get_descriptions(file):

	descriptions_data = dict()

	file = open(file).read()

	descriptions = file.split('\n')

	for description in descriptions:

		if len(description) > 1:

			id_plus_descriptions = description.split('\t')

			idx = id_plus_descriptions[0]

			description = id_plus_descriptions[1]

			descriptions_data[idx] = description

	return descriptions_data

def create_tokenizer(descriptions):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(descriptions)
	return tokenizer

def OHE(index_list,length):

	label = np.zeros(length)

	for index in index_list:
		label[index-1] = 1
	#print(label.shape)
	return label

def conv_2d(x,f,transpose,chanDim):
		if transpose:
			x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
		else:
			x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
		x = LeakyReLU(alpha=0.2)(x)
		x = BatchNormalization(axis=chanDim)(x)

		return x

def keyword_Classifier(base_model,vocab_size):

	for layer in base_model.layers:

		layer.trainable = False
	
	feature  = base_model.layers[-1].output
	feature = Dropout(0.5)(feature)
	feature = Flatten()(feature)
	final_model = Dense(512, activation='relu')(feature)

	final_model = Dense(vocab_size, activation='sigmoid')(final_model)

	model = Model(inputs=base_model.input, outputs=final_model)

	print(model.summary())
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def generator(idx, caption_data, batch_Size,tokenizer, vocab_size):

	IMAGES_PATH = 'C:/Users/mdrah/Downloads/Code_CLEF2018/CaptionTraining2018small'

	X1, y = list(), list()
	count = 0

	while True:
		for id_ in idx:

			path = os.path.join(IMAGES_PATH,id_+'.jpg')
			
			if os.path.exists(path):

				y_ = []

				caption = caption_data[id_]
				new_label = tokenizer.texts_to_sequences([caption])[0]
			
				out_seq = OHE(new_label,vocab_size)

				X = load_image(path,(224,224))
				
					
				X1.append(X)
				y.append(out_seq)
				count+=1
			
			if count == batch_Size:
				count = 0
				yield np.array(X1), np.array(y)
				X1, y = list(), list()

IMAGES_PATH = 'C:/Users/mdrah/Downloads/Code_CLEF2018/CaptionTraining2018small'

#run get_keywords to obtain this file
descriptions_data = get_descriptions('keyword_description.txt')

images_id = list(descriptions_data)

captions = [descriptions_data[description] for description in descriptions_data]

tokenizer = create_tokenizer(captions)

caption_labels = json.loads(tokenizer.get_config()['word_counts'])

vocab_size = len(caption_labels)

base_model = DenseNet121(input_shape=(224,224,3),weights='imagenet')
base_model = Model(inputs=base_model.input,outputs=base_model.layers[-2].output)

model = keyword_Classifier(base_model,vocab_size)

paths = [path.split('\\')[-1].split('.jpg')[0] for path in glob(IMAGES_PATH + '/*') if path.split('\\')[-1].split('.jpg')[0] in images_id]

data_length = len(paths)

train_data = paths[:int(data_length*0.75)]
val_data = paths[int(data_length*0.75):]

EPOCHS = 1000
INIT_LR = 1e-1

model_path = 'caption_model.h5'
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

train_gen = generator(train_data,descriptions_data,2,tokenizer,vocab_size)
val_gen = generator(val_data,descriptions_data,2,tokenizer,vocab_size)

model.fit(train_gen, epochs=EPOCHS,
			verbose=1, steps_per_epoch=len(train_data)//2 ,
			validation_data = val_gen, 
			validation_steps=len(val_data)//2, callbacks=[checkpoint])
