from tensorflow.keras.applications import VGG19,VGG16,InceptionV3,ResNet50,DenseNet121
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv1D, MaxPool2D, UpSampling1D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed
from tensorflow.keras.layers import Embedding, RepeatVector
from tensorflow.keras.layers import Dropout, Flatten, RepeatVector, TimeDistributed, Bidirectional, concatenate, Lambda, dot, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

import tensorflow as tf

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def get_model(model):
	output_layer = -1

	if model in ['VGG16','VGG19','RESNET50',
			'INCEPTION','DENSE121','XCEPTION']:
		if model == 'VGG19':
			model = VGG19(weights='imagenet')
			model = Model(inputs=model.inputs, 
					outputs=model.layers[output_layer].output)
		elif model == 'VGG16':
			model = VGG16(weights='imagenet')
			model = Model(inputs=model.inputs, 
					outputs=model.layers[output_layer].output)
		elif model == 'RESNET50':
			model = ResNet50(weights='imagenet')
			model = Model(inputs=model.inputs, 
					outputs=model.layers[output_layer].output)
		elif model == 'DENSE121':
			model = DenseNet121(weights='imagenet')
			model = Model(inputs=model.inputs, 
					outputs=model.layers[output_layer].output)
		elif model == 'INCEPTION':
			model = InceptionV3(weights='imagenet')
			model = Model(inputs=model.inputs, 
					outputs=model.layers[output_layer].output)
		elif model == 'XCEPTION':
			model = InceptionV3(weights='imagenet')
			model = Model(inputs=model.inputs, 
				outputs=model.layers[output_layer].output)
		return model
	else:
		raise()



def v_res(model_name = None):
	# initialize the input shape to be "channels last" along with
	# the channels dimension itself
	# channels dimension itself
	base_model = get_model(model_name) 

	for layer in base_model.layers:
		layer.trainable = False
	
	x = base_model.output
	
	x = Dense(256,activation='relu')(x)
	#x = Dense(256,activation='relu')(x)
	outputs = Dense(60,activation='softmax')(x)

	model = Model(base_model.input, outputs, name="autoencoder")

	return model

def sae(width=None, height=None, depth=None):
	# initialize the input shape to be "channels last" along with
	# the channels dimension itself
	# channels dimension itself
	def conv_2d(x,f,transpose,chanDim):
		if transpose:
			x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
		else:
			x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
		x = LeakyReLU(alpha=0.2)(x)
		x = BatchNormalization(axis=chanDim)(x)

		return x

	# initialize the input shape to be "channels last" along with
	# the channels dimension itself
	# channels dimension itself
	inputShape = (height, width, depth)
	chanDim = -1

	inputs = Input(shape=inputShape)
	x = inputs
	# loop over the number of filters
	x = conv_2d(x,6,0,chanDim)

	x1 = x

	x = conv_2d(x,3,1,chanDim)

	x = conv_2d(x1,12,0,chanDim)
	x2 = x

	x = conv_2d(x,6,1,chanDim)

	x = conv_2d(x2,32,0,chanDim)
	x3 = x

	x = conv_2d(x,12,1,chanDim)

	x = conv_2d(x3,64,0,chanDim)

	x4 = x

	x = conv_2d(x,24,1,chanDim)

	x = conv_2d(x4,128,0,chanDim)

	x = Flatten()(x)
	#x = conv_2d(x,3,1,chanDim)

	outputs = Dense(60,activation='softmax')(x)
	# construct our autoencoder model
	autoencoder = Model(inputs, outputs, name="autoencoder")
	# return the autoencoder model
	#autoencoder.compile(optimizer='adam', loss='mse')

	return autoencoder

def sae2(width=None, height=None, depth=None):
	# initialize the input shape to be "channels last" along with
	# the channels dimension itself
	# channels dimension itself
	def conv_2d(x,f,transpose,chanDim):
		if transpose:
			x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
		else:
			x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
		x = LeakyReLU(alpha=0.2)(x)
		x = BatchNormalization(axis=chanDim)(x)

		return x

	# initialize the input shape to be "channels last" along with
	# the channels dimension itself
	# channels dimension itself
	inputShape = (height, width, depth)
	chanDim = -1

	inputs = Input(shape=inputShape)
	x = inputs
	# loop over the number of filters
	x = conv_2d(x,6,0,chanDim)

	x1 = x

	x = conv_2d(x,3,1,chanDim)

	x = conv_2d(x1,12,0,chanDim)
	x2 = x

	x = conv_2d(x,6,1,chanDim)

	x = Flatten()(x2)
	#x = conv_2d(x,3,1,chanDim)

	outputs = Dense(60,activation='softmax')(x)
	# construct our autoencoder model
	autoencoder = Model(inputs, outputs, name="autoencoder")
	# return the autoencoder model
	#autoencoder.compile(optimizer='adam', loss='mse')

	return autoencoder

def dn(width=None, height=None, depth=None):
	# initialize the input shape to be "channels last" along with
	# the channels dimension itself
	# channels dimension itself
	def conv_2d(x,f,transpose,chanDim):
		if transpose:
			x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
		else:
			x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
		x = LeakyReLU(alpha=0.2)(x)
		x = BatchNormalization(axis=chanDim)(x)

		return x

	# initialize the input shape to be "channels last" along with
	# the channels dimension itself
	# channels dimension itself
	inputShape = (height, width, depth)
	chanDim = -1

	inputs = Input(shape=inputShape)
	x = inputs
	# loop over the number of filters

	x = conv_2d(x,32,0,chanDim)
	x = MaxPool2D(pool_size=(2,2),strides=(1,1),padding="same")(x)

	x = conv_2d(x,64,0,chanDim)
	x = MaxPool2D(pool_size=(2,2),strides=(1,1),padding="same")(x)

	x = conv_2d(x,128,0,chanDim)
	x = MaxPool2D(pool_size=(2,2),strides=(1,1),padding="same")(x)

	x = conv_2d(x,256,0,chanDim)
	x = MaxPool2D(pool_size=(2,2),strides=(1,1),padding="same")(x)

	x = Flatten()(x)
	
	#outputs = Dense(128,activation='relu')(x)
	outputs = Dense(60,activation='softmax')(x)
	# construct our autoencoder model
	autoencoder = Model(inputs, outputs, name="autoencoder")
	# return the autoencoder model
	#autoencoder.compile(optimizer='adam', loss='mse')

	return autoencoder

m = dn(32,32,3)
print(m.summary())
