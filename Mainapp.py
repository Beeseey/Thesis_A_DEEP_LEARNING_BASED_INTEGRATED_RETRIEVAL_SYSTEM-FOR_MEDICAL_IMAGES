from PyQt5 import QtCore, QtGui, QtWidgets
#from PyQt5.QtWidgets import
from GUI import gui
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_ 
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import load_model, Model
import numpy as np
import sys
import os
import cv2
import imutils
from skimage.transform import resize
from PIL import ImageFile,Image
import functools
import pickle
import json
#from run import load_data, restructure_results
#from Retrieval.search import get_results
#from extract import extract_
import time
import tensorflow as tf
#import tensorflow as tf
from scipy.spatial import distance

from cbir_util import get_res

FILEPATH = ''
IMAGE = None
#QueryImage = None
def load_data(data_file):
	if 'pkl' in data_file:
		data = pickle.load(open(data_file, 'rb'))
	elif 'json' in data_file:
		data = json.load(open(data_file, 'r'))
	return data

def selectFile():
	global FILEPATH
	
	FILEPATH = QtWidgets.QFileDialog.getOpenFileName()[0]
	if os.path.exists(FILEPATH):
		load_image_in_label(FILEPATH)

def MSE(x,y):
	MSE = np.square(np.subtract(x,y)).mean()

	return MSE

def EUC(x,y):
	EUC = np.linalg.norm(x-y)

	return EUC

def COD(x,y):
	COD = distance.correlation(x,y)

	return COD

def CTB(x,y):
	manhattan_distance = distance.cityblock(x, y)/len(x)

	return manhattan_distance



def load_image(mode, path, target_size):
	w,h = 32,32

	x = [0, 0, 0, 0, 0, 0, 0, 32, 32, 32, 32, 32, 32, 32, 
		64, 64, 64, 64, 64, 64, 64, 96, 96, 96, 96, 96, 96, 96, 
		128, 128, 128, 128, 128, 128, 128, 160, 160, 160, 160, 160, 160, 160, 
		192, 192, 192, 192, 192, 192, 192]
	y = [0, 32, 64, 96, 128, 160, 192, 0, 32, 64, 96, 128, 160, 192,
		0, 32, 64, 96, 128, 160, 192, 0, 32, 64, 96, 128, 160, 192,
		0, 32, 64, 96, 128, 160, 192, 0, 32, 64, 96, 128, 160, 192,
		0, 32, 64, 96, 128, 160, 192]
	
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
	if mode == 1 or mode == 2:
		print('preprocess_done')
		image = preprocess_input(image)
		image = np.expand_dims(image,axis=0)
	if mode == 3:
		image = preprocess_input(image)
		image = [image[y[i]:y[i]+h, x[i]:x[i]+w] for i in range(49)]

	return image

def pad(seq,count):

	for i in range(1,7-count):
		seq[0].insert(0,0)
	return np.array(seq)

def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def get_keywords(path, model):
	
	image = load_image(2,path,(224,224))

	pred = []

	seq = [1]

	for i in range(7):

		sequence = []

		#sequence = sequence.tolist()
		sequence.append(pad([seq+pred],i)[0])
		word = model.predict((image,np.array(sequence)))
		pred.append(np.argmax(word)+1)
	return pred

def get_captions():

	global FILEPATH, tokenizer, caption_data

	

	model = 'C:/Users/mdrah/Downloads/Code_CLEF2018/Thesis/lstm_caption_model.h5'
	model = load_model(model)

	key_words = get_keywords(FILEPATH,model)

	key_words_disp = np.unique(np.array(key_words))

	print(key_words)

	key_words_disp = key_words_disp.tolist()

	words = [word_for_id(inte,tokenizer) for inte in key_words_disp]

	text = 'GENERATED CAPTIONS \n\n'+'\n'.join(words)

	result_label = UI.result_widget
	result_label.clear()

	result_label.setText(text)

	return key_words


def get_cbir_results(mode,search_number,dist):

	global cbir_tree, cbir_data

	'''

	results = get_res(FILEPATH,cbir_data,cbir_tree,search_number)
	'''


	model = DenseNet121(input_shape=(224,224,3),weights='imagenet')

	model = Model(inputs=model.inputs, 
			outputs=model.layers[-2].output)

	cbir_image = load_image(mode,FILEPATH,(224,224))
	prediction = model.predict(cbir_image)



	result = cbir_tree.get_n_nearest_neighbors(prediction,50)

	#print(result)

	results = [[os.path.join('C:/Users/mdrah/Downloads/ImageClef 2013',cbir_data[tuple(res[1])]),res[0]] for res in result]

	results.sort(key=lambda x: x[1])
	

	#print(results)
	return results

def clean_res(res):
	r = sorted(res[0].tolist())

	r = [i for i in r if i != 0]

	return tuple(r)

def get_tbir_results(mode,search_number,dist):
	from tensorflow.keras.preprocessing.sequence import pad_sequences

	res = []

	pred = get_captions()

	with open('C:/Users/mdrah/Downloads/ImageClef 2013/trees/CTB_tbir_tree.pkl', 'rb') as json_file:
		caption_tree = pickle.load(json_file)

	X = np.array(pad_sequences([sorted(pred)],7))
	results = caption_tree.get_n_nearest_neighbors(X,search_number)

	for result in results:
		res_list = caption_data[clean_res(result[1])]
		if isinstance(res_list,str):
			res_list = [res_list]
		for res_path in res_list:
			res.append([res_path,result[0]])
			if len(res) == search_number:
				break
		if len(res) == search_number:
				break
	return res

def count_label(labels):
	
	new_labels = np.zeros(60).tolist()


	for label_index in labels:
		new_labels[label_index]+=1
	return new_labels

def get_visual_words():

	model_name = 'C:/Users/mdrah/Downloads/ImageClef 2013/dn_4_ep1000_lr0.001_mse_0.89.h5'
	model = load_model(model_name)

	image = load_image(3,FILEPATH,(224,224))

	label = np.argmax(model.predict(np.array(image)),axis=-1).tolist()

	label = count_label(label)
	text = ''

	groups = ['angio_coronary', 'background_black', 'background_blue', 'background_brown', 'background_green_blue', 'background_grey', 
		'background_white', 'background_yellow', 'charts_bar_color', 'charts_bar_grey', 
		'chemical_structure_bw', 'chemical_structure_color', 'ct_abdomen_liver', 'ct_abdomen_spine', 
		'ct_bronchi', 'ct_corner_black_grey', 'ct_fat_tissue_corner', 'ct_ftissue_white', 'ct_grey_fattissue', 
		'ct_groundglass', 'ct_honeycomb', 'ct_lungcyst', 'ct_nodules', 
		'ct_tissue_normal', 'dental_white', 'ecg_signal', 'endoscopy_red', 'gel', 
		'handdrawn_color', 'handdrawn_grey', 'microscopy_cell_blue', 'microscopy_cell_pink', 
		'microscopy_electron_grey', 'microscopy_fluorescence', 'microscopy_histo_blue', 'microscopy_light_green', 
		'microscopy_light_pink', 'microscopy_transmission_grey1', 'microscopy_transmission_grey2', 
		'microscopy_violet', 'mri_fetus', 'mri_head_brain', 'mri_leg', 
		'organ_tissue_other', 'organ_tissue_red', 'pet_ct_color', 'pet_lung_cancer', 'photo_eye', 
		'photo_tissue_cardiac', 'print_letters_black_bkg', 'print_letters_white_bkg', 
		'skin_abnormal', 'skin_normal_dark', 'skin_normal_light', 'us_color', 'us_grey', 
		'xray_chest_bone', 'xray_finger_bone', 'xray_knee', 'xray_vertebra']
	
	for g,c in zip(groups,label):
		text= text+g+'  '+str(c)+'\n'
	text = 'GENERATED VISUAL WORDS\n'+text

	result_label = UI.result_widget
	result_label.clear()

	result_label.setText(text)

	return label



def get_pbir_results(mode,search_number,dist):

	global pbir_data

	res = []

	with open('C:/Users/mdrah/Downloads/ImageClef 2013/trees/EUC_pbir_tree.pkl', 'rb') as json_file:
		pbir_tree = pickle.load(json_file)
	
	

	label = get_visual_words()
	

	results = pbir_tree.get_n_nearest_neighbors(label, search_number)

	for result in results:

		res.append([pbir_data[tuple(result[1].tolist())],result[0]])

	return res



def get_search_result(search_number,mode):
	global UI,FILEPATH

	mode_select_dict = {1:'cbir',2:'tbir',3:'pbir'}
	if UI.mse_button.isChecked():
		dist = 'JAC'
	elif UI.euc_button.isChecked():
		dist = 'EUC'
	elif UI.cod_button.isChecked():
		dist = 'CTB'
	else:
		dist = None

	if mode == 1:
		results = get_cbir_results(1,search_number,dist)
	elif mode == 2:
		results = get_tbir_results(2,search_number,dist)
	elif mode == 3:
		results = get_pbir_results(3,search_number,dist)

	return results

def search(mode):
	global UI,FILEPATH

	if UI.concept_ret_button.isChecked() and mode == 2:
		#mode = 1
		search_box = UI.patch_search_number
	elif UI.cbir_button.isChecked() and mode == 1:
		#mode = 2
		search_box = UI.cbir_search_number
	elif UI.patch_button.isChecked() and mode == 3:
		#mode = 3
		search_box = UI.concept_search_number
	else:
		search_box = None

	if os.path.exists(FILEPATH):
		
		if 	search_box is None:
			search_number = 10
		else:
			if search_box.text() == '':
				search_number = 10
			else:
				search_number = int(search_box.text())

		result = get_search_result(search_number,mode)
		print(result)
		display_search(search_number,result)
	#print("Good",mode)
def combine_res():

	global UI,FILEPATH

	search_number_1 = int(UI.cbir_search_number.text())
	search_number_2 = int(UI.concept_search_number.text())

	if UI.concept_ret_button.isChecked() and UI.cbir_button.isChecked():
		result_1 = get_search_result(search_number_1,1)
		result_2 = get_search_result(search_number_2,2)
	result_1.extend(result_2)

	display_search(len(result_1),result_1)

def clear_r():
	global UI

	result_label = UI.result_widget
	result_label.clear()



def view_large_scale(event,path):
	global UI

	image = cv2.imread(path)
	width,height = image.shape[1],image.shape[0]
	
	pop = QtWidgets.QLabel()
	UI.f.append(pop)
	pop.setFixedSize(width,height)
	pop.move(300+(len(UI.f)*10),300+(len(UI.f)*10))
	pop.setAlignment(QtCore.Qt.AlignCenter)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = QtGui.QImage(image.tobytes(), image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
	pix = QtGui.QPixmap.fromImage(image)
	pop.setPixmap(pix)
	pop.show()

	result_label = UI.result_widget
	print(result_label.frameGeometry().width())
	print(len(" fjlgfljg klg ukkjfs ksfkjdhkjgh.jkdh.,jgh.kdjh d "))
	result_label.clear()
	#result_label.setText('hjvdkhfvjvfhsfgjsg \n fjlgfljg klg ukkjfs ksfkjdhkjgh.jkdh.,jgh.kdjh d \n khfuhf fik hoihihliuhivu hiv iuvi hvvih hvibibhiuhb')
	result_label.setText('SEARCH RESULTS: \n COVID 19: 10 out of 10 images \n Pnuemonia: 0 out of 10 images \n No Findings: 0 out of 10 images'\
							'\n Atelectasis: 0 out of 10 images \n Cardiomegaly: 0 out of 10 images \n Hernia: 0 out of 10 images'\
							'\n Fibrosis: 0 out of 10 images \n Emphysema: 0 out of 10 images \n Edema: 0 out of 10 images'\
							'\n Effusion: 0 out of 10 images \n Nodule: 0 out of 10 images \n Mass: 0 out of 10 images'\
							'\n Infiltration: 0 out of 10 images \n Pneumothorax: 0 out of 10 images \n Pleural_Thickening: 0 out of 10 images')

def display_search(number,results):
	global UI,result_layout

	#print(results)


	query_result_label = UI.search_results_widget
	cl = UI.clearance
	search_result_width = UI.search_result_width
	count = 0
	width = query_result_label.frameGeometry().width()
	height = query_result_label.frameGeometry().height()

	queue_label = QtWidgets.QLabel(query_result_label)
	queue_label.setGeometry(QtCore.QRect(0, 0, width, height))
	queue_label.show()

	if result_layout is not None:
		for i in reversed(range(result_layout.count())): 
			result_layout.itemAt(i).widget().setParent(None)
	
	queue_label_scroll = QtWidgets.QLabel(query_result_label)


	if number%4 > 0:
		col_num = number//4+1
	elif number%4 == 0:
		col_num = number//4
	
	queue_label_scroll.setGeometry(QtCore.QRect(0, 0, 
		(search_result_width*4)+(cl*4), (search_result_width*col_num)+(cl*col_num)))
	queue_label_scroll.setObjectName("queue_label_scroll")

	scrollArea = QtWidgets.QScrollArea()
	scrollArea.setWidgetResizable(False)
	scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
	scrollArea.setWidget(queue_label_scroll)

	verticalLayout = QtWidgets.QVBoxLayout(queue_label)
	verticalLayout.addWidget(scrollArea)
	verticalLayout.setContentsMargins(0, 0, 0, 0)
	verticalLayout.setObjectName("verticalLayout")

	result_layout = verticalLayout

	for i in range(number):
		for j in range(4):
			result = results[count]
			#print(resuls)
			path,score = result
			count+=1
			each_result_Label = QtWidgets.QLabel(queue_label_scroll)
			x = (search_result_width*j)+(cl*j)
			y = (search_result_width*i)+(cl*i)
			each_result_Label.setStyleSheet("background-color:rgb(230, 230, 230);font-size:35px;color:white")
			
			each_result_Label.setGeometry(QtCore.QRect(x,y,search_result_width,search_result_width))
			each_result_Label.setAlignment(QtCore.Qt.AlignCenter)
			each_result_Label.show()
			display_image_in_label(path,each_result_Label)
			each_result_Label.mousePressEvent = functools.partial(view_large_scale,path=path)
			#each_result_Label.mousePressEvent = view_large_scale(path=path)
			if count == number:
				break
		if count == number:
			break
	#each_result_Label.mousePressEvent = functools.partial(selectFile,source_object=each_result_Label)
	

	
def display_image_in_label(FILEPATH,label):

	width = label.frameGeometry().width()
	height = label.frameGeometry().height()
	IMAGE = cv2.imread(FILEPATH)
	image = IMAGE
	dim = (width - width%100,height-height%100)
	image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	img = QtGui.QImage(image.tobytes(), image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
	pix = QtGui.QPixmap.fromImage(img)

	label.setPixmap(pix)



def load_image_in_label(FILEPATH):
	global UI,IMAGE

	query_label = UI.query_image
	width = query_label.frameGeometry().width()
	height = query_label.frameGeometry().height()
	IMAGE = cv2.imread(FILEPATH)
	image = IMAGE
	dim = (width - width%100,height-height%100)
	image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	img = QtGui.QImage(image.tobytes(), image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
	pix = QtGui.QPixmap.fromImage(img)

	query_label.setPixmap(pix)

if __name__ == "__main__":
	print('Loading')

	gpus= tf.config.experimental.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(gpus[0], True)

	with open('C:/Users/mdrah/Downloads/ImageClef 2013/CTB_cbir_tree_2.pkl', 'rb') as json_file:
		cbir_tree = pickle.load(json_file)
	with open('C:/Users/mdrah/Downloads/ImageClef 2013/auto_enc_features_2.pkl', 'rb') as json_file:
		cbir_data = pickle.load(json_file)
	
	with open('C:/Users/mdrah/Downloads/ImageClef 2013/restructured_data_captions.pkl', 'rb') as json_file:
		caption_data = pickle.load(json_file)

	with open('C:/Users/mdrah/Downloads/ImageClef 2013/restructured_data.pkl', 'rb') as json_file:
		pbir_data = pickle.load(json_file)

	_, tokenizer = load_data('C:/Users/mdrah/Downloads/ImageClef 2013/parallel_info.pkl')
	

	app = QtWidgets.QApplication(sys.argv)
	MainWindow = QtWidgets.QMainWindow()
	screen_resolution = app.desktop().screenGeometry()

	app.setStyleSheet("QPushButton{color: black;border-style: outset;"
				"border-width: 3px;border-radius: 10px;border-color: rgb(200,200,200);}"
				"QPushButton:pressed{border-width: 0px}")

	width, height = screen_resolution.width(), screen_resolution.height()
	UI = gui.Ui_MainWindow()
	UI.setupUi(MainWindow,width,height)
	result_layout=None
	UI.select_button.clicked.connect(selectFile)
	UI.cbir_search_button.clicked.connect(lambda: search(1))
	UI.patch_search_button.clicked.connect(lambda: search(3))
	UI.concept_search_button.clicked.connect(lambda: search(2))

	UI.get_concept_button.clicked.connect(get_captions)

	UI.get_patches_button.clicked.connect(get_visual_words)

	UI.combine_button.clicked.connect(combine_res)

	UI.search_number_c.clicked.connect(clear_r)
	
	MainWindow.show()
	sys.exit(app.exec_())