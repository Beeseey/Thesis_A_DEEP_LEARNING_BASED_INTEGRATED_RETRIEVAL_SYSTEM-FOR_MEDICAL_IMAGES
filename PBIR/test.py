import os
from glob import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report
import numpy as np
import cv2
from PIL import ImageFile,Image

ImageFile.LOAD_TRUNCATED_IMAGES = True 

DATASET = 'test'
dataset_groups = [os.path.split(group)[-1] for group in glob(DATASET+'/*')]

print(dataset_groups)

raise()

for path in glob('test'+'/*/*'):
	if 'Thumbs' in path:
		os.system('rm '+ path)


model_name = 'dn_5_ep1000_lr0.001_mse.h5'
model = load_model(model_name)

y_true = []
y_pred = []

def OHE(index,length):

	label = np.zeros(length)

	label[index] = 1

	return label

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
	image = np.expand_dims(image,axis=0)
	
	#image = image.astype("float32")/255.0
	return image

test_data = glob('test'+'/*/*')


count = 0

for path in test_data:

	count+=1
	print(count,'of',len(test_data))

	label = os.path.split(path)[-2].split('\\')[-1]
	y = dataset_groups.index(label)

	y_true.append(y)

	X = load_image(path,(32,32))
	y_ = model.predict(X)

	y_pred.append(np.argmax(y_))
	
#print(y_pred,y_true)
print(classification_report(y_true, y_pred, target_names=dataset_groups))