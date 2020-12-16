from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D, AveragePooling2D, Conv1D, Conv2DTranspose
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout, Flatten, RepeatVector, TimeDistributed, Bidirectional, concatenate, Lambda, dot, Activation, add
from tensorflow.keras.layers import add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Reshape
from tensorflow.keras import backend as K
from tensorflow.keras.applications import DenseNet121
import numpy as np


def conv_2d(x,f,transpose,chanDim):
		if transpose:
			x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
		else:
			x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
		x = LeakyReLU(alpha=0.2)(x)
		x = BatchNormalization(axis=chanDim)(x)

		return x

def auto_encoder_image():

	base_model = DenseNet121(input_shape=(224,224,1),weights='imagenet', include_top=True)

	for layer in base_model.layers:

		layer.trainable = False
	
	feature  = base_model.layers[64].output


	#feature = conv_2d(feature,1024,-1)

	feature = conv_2d(feature,32,0,-1)

	volumeSize = K.int_shape(feature)

	feature = Flatten()(feature)

	feature = Dense(512, activation='relu')(feature)

	feature = Dense(np.prod(volumeSize[1:]))(feature)
	
	feature = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(feature)

	feature = conv_2d(feature,8,1,-1)
	feature = conv_2d(feature,16,1,-1)
	feature = conv_2d(feature,32,1,-1)
	feature = conv_2d(feature,64,1,-1)

	outputs = Conv2DTranspose(1, (1, 1), padding="same")(feature)
	outputs = Activation("sigmoid", name="decoded")(outputs)

	model = Model(inputs=base_model.input, outputs=outputs)

	print(model.summary())
	model.compile(loss='mse', optimizer='rmsprop')
	return model


def auto_encoder(width=None, height=None, depth=None, filters=None, 
	latentDim=None):
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
	base_model = DenseNet121(input_shape=(224,224,3),weights='imagenet', include_top=True)

	for layer in base_model.layers:

		layer.trainable = False
	
	feature  = base_model.layers[-2].output
	#feature = Dense(512, activation='relu')(feature)
	inputShape = (height, width, depth)
	chanDim = -1
	# define the input to the encoder
	inputs = base_model.input
	x = inputs
	# loop over the number of filters
	x = conv_2d(x,8,0,chanDim)
	x1 = x
	x = conv_2d(x,16,0,chanDim)
	x2 = x
	x = conv_2d(x,32,0,chanDim)
	x3 = x
	# flatten the network and then construct our latent vector
	volumeSize = K.int_shape(x)
	x = Flatten()(x)
	latent = Dense(1024, name="encoded")(x)
	latent = add([feature,latent])
	x = Dense(np.prod(volumeSize[1:]))(latent)
	x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
	# loop over our number of filters again, but this time in
	# reverse order
	x = concatenate([x,x3])
	x = conv_2d(x,32,1,chanDim)
	x = concatenate([x,x2])
	x = conv_2d(x,16,1,chanDim)
	x = concatenate([x,x1])
	x = conv_2d(x,3,1,chanDim)

	#x = conv_2d(x,3,1,chanDim)
	
	#x = concatenate([x,inputs])
	outputs = Conv2DTranspose(3, (3, 3), padding="same")(x)
	outputs = Activation("sigmoid", name="decoded")(x)
	# construct our autoencoder model
	autoencoder = Model(inputs, outputs, name="autoencoder")
	# return the autoencoder model
	autoencoder.compile(optimizer='adam', loss='mse')

	return autoencoder

'''
model = auto_encoder(224,224,3)
print(model.summary())
raise()
'''