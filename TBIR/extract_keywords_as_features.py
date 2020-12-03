from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2
import numpy as np
from skimage.transform import resize
from PIL import ImageFile,Image
from glob import glob
from tqdm import tqdm
import pickle, json
import os

import tensorflow as tf

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

model = 'caption_model.h5'
model = load_model(model)

def load_data(data_file):
    if 'pkl' in data_file:
        data = pickle.load(open(data_file, 'rb'))
    elif 'json' in data_file:
        data = json.load(open(data_file, 'r'))
    return data

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r', encoding='ISO-8859-1')
    # read all text
    text = file.read()
    # close the file
    file.close()

    return text

def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        
        # split id from description
        if len(line) > 1:
            tokens = line.split('\t')
            image_id, image_desc = tokens[0], tokens[1]
            # skip images not in the set
            if image_id in dataset:
                # create list
                if image_id not in descriptions:
                    descriptions[image_id] = list()
                # wrap description in tokens
                desc = 'startseq ' + image_desc + ' endseq'
                # store
                descriptions[image_id].append(desc)
    return descriptions

def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def load_image(path, target_size):
    
    try:
        image = load_img(path, target_size=target_size)
        # convert the image pixels to a numpy array
        image = img_to_array(image)
    except:
        image = cv2.imread(path)
        image =  cv2.resize(image, target_size, interpolation = cv2.INTER_AREA)
        image = Image.fromarray(image)
        image = img_to_array(image)
    image = preprocess_input(image)

    return image

paths = 'C:/Users/mdrah/Downloads/ImageClef 2013/images/*'


path_list = [path for path in glob(paths)]

tokenizer = load_data('tokenizer.pkl')

vocab_size = len(tokenizer.word_index) + 1

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def pad(seq,count):

    for i in range(1,7-count):
        seq[0].insert(0,0)
    return np.array(seq)

data = dict.fromkeys(path_list)

maxlen = 7

for i in tqdm(range(0,len(path_list),5)):

    path = path_list[i:i+5]

    seq_ = [[1],[1],[1],[1],[1]]

    p_text = ''
    images = []

    for each_path in path:
        image = load_image(each_path,(224,224))
        images.append(image)

    pred = [[],[],[],[],[]]

    words = []



    for i in range(maxlen):
        sequence = []
        count = 0
        for seq,pred_ in zip(seq_,pred):

            sequence.append(pad([seq+pred_],i)[0])
        
        word = model.predict([np.array(images),np.array(sequence)])

        for w,p in zip(word,pred):
            p.append(np.argmax(w)+1)

    for each_pred,each_path in zip(pred,path):
        data[each_path] = each_pred

def write_json(data, filename): 
    with open(filename,'wb') as f: 
        pickle.dump(data, f) 

write_json(data,'caption_features.pkl')