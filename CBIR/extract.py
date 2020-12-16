from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model
import cv2
import numpy as np
from skimage.transform import resize
from PIL import ImageFile,Image
from glob import glob
from tqdm import tqdm
import pickle
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
import tensorflow as tf

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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
    #image = np.expand_dims(image,axis=0)

    return image

model_name = 'modal_class.h5'
base_model = load_model(model_name)
model = Model(inputs=base_model.inputs, 
            outputs=base_model.get_layer('encoded').output)
'''
model = DenseNet121(input_shape=(224,224,3),weights='imagenet')

model = Model(inputs=model.inputs, 
			outputs=model.layers[69].output)
'''

paths = 'images/*'
path_list = [path for path in glob(paths)]

data = {}

for i in tqdm(range(0,len(path_list),20)):

    paths = path_list[i:i+20]

    images = []
    for path in paths:
        images.append(load_image(path,(224,224)))

    images = np.array(images)
    features = model.predict(images)

    for path,feature in zip(paths,features):
        data[tuple(feature)] = path

def write_json(data, filename): 
    with open(filename,'wb') as f: 
        pickle.dump(data, f) 
#print(new_data)
write_json(data,'modal_class_feature.pkl')