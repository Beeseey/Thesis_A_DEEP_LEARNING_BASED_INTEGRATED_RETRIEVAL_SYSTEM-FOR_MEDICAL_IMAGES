from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D, AveragePooling2D, Conv1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout, Flatten, RepeatVector, TimeDistributed, Bidirectional, concatenate, Lambda, dot, Activation
from tensorflow.keras.layers import add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.applications import DenseNet121

def Annotation_model(vocab_size, max_len):

	base_model = DenseNet121(input_shape=(224,224,3),weights='imagenet')

	for layer in base_model.layers:

		layer.trainable = False
	
	feature  = base_model.layers[-2].output

	final_model = Dense(vocab_size, activation='sigmoid')(feature)

	model = Model(inputs=base_model.input, outputs=final_model)

	print(model.summary())
	
	return model