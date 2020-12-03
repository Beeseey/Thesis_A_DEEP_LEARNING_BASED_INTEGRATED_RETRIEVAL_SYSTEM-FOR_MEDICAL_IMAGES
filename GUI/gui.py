import sys
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
	def setupUi(self, MainWindow, s_width, s_height):
		_translate = QtCore.QCoreApplication.translate
		window_width = s_width*0.6
		window_height = s_height*0.65
		w = window_width
		h = window_height
		#print(w,h)
		MainWindow.setObjectName("MainWindow")
		MainWindow.setFixedSize(w,h)

		self.f = []

		self.clearance = w*0.015
		cl = self.clearance

		query_widget_width, query_widget_height = (w - (cl*2))*0.7, (h - (cl*3)) * 0.25
		result_widget_width, result_widget_height = w-query_widget_width-(cl*3), h - (cl*2)
		search_widget_width, search_widget_height = query_widget_width, (h - (cl*3)) * 0.75
		query_image_width, query_image_height = (query_widget_width - cl)*0.25, query_widget_height
		query_selection_width, query_selection_height = (query_widget_width - cl)*0.75, query_widget_height
		button_height = cl*1.5
		button_width = cl*5.5
		text_box_height = cl*1.2
		url_box_width = query_selection_width - cl - button_width

		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setObjectName("centralwidget")
		self.centralwidget.setGeometry(0,0,w,h)
		self.centralwidget.setStyleSheet("background-color:rgb(230, 230, 230)")

		self.query_widget = QtWidgets.QLabel(self.centralwidget)
		self.query_widget.setGeometry(QtCore.QRect(cl, cl, 
							query_widget_width, query_widget_height))
		self.query_widget.setObjectName("query_widget")
		self.query_widget.setStyleSheet("background-color:rgb(230, 230, 230);font-size:35px;color:white")
		self.query_widget.setAlignment(QtCore.Qt.AlignCenter)

		self.result_widget = QtWidgets.QLabel(self.centralwidget)
		self.result_widget.setGeometry(QtCore.QRect(query_widget_width+(cl*2), cl, 
							result_widget_width, result_widget_height))
		self.result_widget.setObjectName("result_widget")
		self.result_widget.setStyleSheet("background-color:rgb(250, 250, 250);font-size:30px;"
											"color:rgb(80,80,80);padding:10 10")
		self.result_widget.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

		self.search_widget = QtWidgets.QLabel(self.centralwidget)
		self.search_widget.setGeometry(QtCore.QRect(cl, query_widget_height+(cl*2), 
							search_widget_width, search_widget_height))
		self.search_widget.setObjectName("search_widget")
		self.search_widget.setStyleSheet("background-color:rgb(250, 250, 250);font-size:35px;color:white")
		self.search_widget.setAlignment(QtCore.Qt.AlignCenter)

		self.search_results_widget = QtWidgets.QLabel(self.search_widget)
		self.search_results_widget.setGeometry(QtCore.QRect(cl, cl, 
							search_widget_width-cl, search_widget_height))
		self.search_results_widget.setObjectName("search_widget")
		self.search_results_widget.setStyleSheet("background-color:rgb(250, 250, 250);font-size:35px;color:white")
		self.search_results_widget.setAlignment(QtCore.Qt.AlignCenter)

		self.search_result_width = ((search_widget_width-cl)-(cl*4))/4

		self.query_image = QtWidgets.QLabel(self.query_widget)
		self.query_image.setGeometry(QtCore.QRect(0, 0, 
							query_image_width, query_image_height))
		self.query_image.setObjectName("search_widget")
		self.query_image.setStyleSheet("background-color:rgb(250, 250, 250);font-size:35px;color:white")
		self.query_image.setAlignment(QtCore.Qt.AlignCenter)



		self.query_selection = QtWidgets.QLabel(self.query_widget)
		self.query_selection.setGeometry(QtCore.QRect(query_image_width+cl, 0, 
							query_selection_width, query_selection_height))
		self.query_selection.setObjectName("search_widget")
		self.query_selection.setStyleSheet("background-color:rgb(230, 230, 230);font-size:35px;color:white")
		#self.query_selection.setAlignment(QtCore.Qt.AlignCenter)

		
		self.combine_button = QtWidgets.QPushButton(self.query_selection)
		self.combine_button.setGeometry(QtCore.QRect(query_selection_width - button_width,
			query_selection_height-button_height,
			button_width,button_height))
		self.combine_button.setStyleSheet("background-color:rgb(210, 210, 210);color:rgb(50, 50, 50);font-size:30px")
		self.combine_button.setText(_translate("MainWindow", "Combine"))
	

		self.select_button = QtWidgets.QPushButton(self.query_selection)
		self.select_button.setGeometry(QtCore.QRect(0,0,
			button_width,button_height))
		self.select_button.setStyleSheet("background-color:rgb(210, 210, 210);color:rgb(50, 50, 50);font-size:30px")
		self.select_button.setText(_translate("MainWindow", "Select..."))


		self.download_button = QtWidgets.QPushButton(self.query_selection)
		self.download_button.setGeometry(QtCore.QRect(0,button_height+(cl/3),
			button_width,button_height))
		self.download_button.setStyleSheet("background-color:rgb(210, 210, 210);color:rgb(50, 50, 50);font-size:30px")
		self.download_button.setText(_translate("MainWindow", "Download"))

		search_download_y = query_selection_height-button_height*2-cl

		radio_selection_y = (search_download_y-(cl/2))/3

		method_label_height = query_selection_height - button_height*2.5
		method_label_width = query_selection_width-(cl*0.5+button_width)

		


		self.method_label = QtWidgets.QLabel(self.query_selection)
		self.method_label.setGeometry(QtCore.QRect(0, 
							query_selection_height-radio_selection_y-button_height*2-cl, 
							method_label_width,method_label_height))
		self.method_label.setObjectName("search_widget")
		self.method_label.setStyleSheet("background-color:rgb(230, 230, 230);font-size:30px;color:rgb(50, 50, 50);"
										"border-style: solid;border-width: 0px;border-radius: 15px;border-color: rgb(50,50,50)")
		self.method_label.setAlignment(QtCore.Qt.AlignCenter)

		self.patch_label = QtWidgets.QLabel(self.method_label)
		self.patch_label.setGeometry(QtCore.QRect(0,0,  
							method_label_width/3,method_label_height))
		self.patch_label.setObjectName("search_widget")
		self.patch_label.setStyleSheet("background-color:rgb(230, 230, 230);font-size:30px;color:rgb(50, 50, 50);"
										"border-style: solid;border-width: 1px;border-radius: 15px;border-color: rgb(50,50,50)")
		self.patch_label.setAlignment(QtCore.Qt.AlignCenter)

		method_name_width = (method_label_width/3)-2

		self.patch_button = QtWidgets.QCheckBox(self.patch_label)
		self.patch_button.setGeometry(QtCore.QRect(1,1,method_name_width,radio_selection_y-2))
		self.patch_button.setStyleSheet("padding-left:10px;border-style: none;")
		self.patch_button.setText(_translate("MainWindow", "Patch Retrieval"))

		self.get_patches_button = QtWidgets.QPushButton(self.patch_label)
		self.get_patches_button.setGeometry(QtCore.QRect((method_label_width/3 - button_width*1.5)/2,radio_selection_y+cl/5,
			button_width*1.5,button_height))
		self.get_patches_button.setStyleSheet("background-color:rgb(210, 210, 210);color:rgb(50, 50, 50);font-size:30px;border-style: outset;")
		self.get_patches_button.setText(_translate("MainWindow", "Get_Patches"))

		self.patch_search_button = QtWidgets.QPushButton(self.patch_label)
		self.patch_search_button.setGeometry(QtCore.QRect(cl/3,(radio_selection_y*2)+(cl*2)/5,
			button_width,button_height))
		self.patch_search_button.setStyleSheet("background-color:rgb(210, 210, 210);color:rgb(50, 50, 50);font-size:30px;border-style: outset;")
		self.patch_search_button.setText(_translate("MainWindow", "Search"))

		self.patch_search_number = QtWidgets.QLineEdit(self.patch_label)
		self.patch_search_number.setGeometry(QtCore.QRect((cl*2)/3+button_width,(radio_selection_y*2)+(cl*2)/5,
				(method_label_width/3)-cl-button_width,button_height))
		self.patch_search_number.setStyleSheet("background-color:rgb(250, 250, 250);color:rgb(50, 50, 50)"
			";border-radius: 10px;font-size:30px;padding-left:10px")
		self.patch_search_number.setPlaceholderText("10")


		#self.vgg_button.toggled.connect(self.check_cboxes)

		self.cbir_label = QtWidgets.QLabel(self.method_label)
		self.cbir_label.setGeometry(QtCore.QRect(method_label_width/3,0,  
							method_label_width/3,method_label_height))
		self.cbir_label.setObjectName("search_widget")
		self.cbir_label.setStyleSheet("background-color:rgb(230, 230, 230);font-size:30px;color:rgb(50, 50, 50);"
										"border-style: solid;border-width: 1px;border-radius: 15px;border-color: rgb(50,50,50)")
		self.cbir_label.setAlignment(QtCore.Qt.AlignCenter)

		self.cbir_button = QtWidgets.QCheckBox(self.cbir_label)
		self.cbir_button.setGeometry(QtCore.QRect(1,1,method_name_width,radio_selection_y-2))
		self.cbir_button.setStyleSheet("padding-left:10px;border-style: none;")
		self.cbir_button.setText(_translate("MainWindow", "Image Retrieval"))

		self.cbir_search_button = QtWidgets.QPushButton(self.cbir_label)
		self.cbir_search_button.setGeometry(QtCore.QRect((method_label_width/3 - button_width*1.5)/2,radio_selection_y+cl/5,
			button_width*1.5,button_height))
		self.cbir_search_button.setStyleSheet("background-color:rgb(210, 210, 210);color:rgb(50, 50, 50);font-size:30px;border-style: outset;")
		self.cbir_search_button.setText(_translate("MainWindow", "Search"))

		self.cbir_search_number = QtWidgets.QLineEdit(self.cbir_label)
		self.cbir_search_number.setGeometry(QtCore.QRect(((method_label_width/3)-((method_label_width/3)-cl-button_width))/2,(radio_selection_y*2)+(cl*2)/5,
				(method_label_width/3)-cl-button_width,button_height))
		self.cbir_search_number.setStyleSheet("background-color:rgb(250, 250, 250);color:rgb(50, 50, 50)"
			";border-radius: 10px;font-size:30px;padding-left:10px")
		self.cbir_search_number.setPlaceholderText("10")


		self.concept_ret_label = QtWidgets.QLabel(self.method_label)
		self.concept_ret_label.setGeometry(QtCore.QRect((method_label_width*2)/3,0,  
							method_label_width/3,method_label_height))
		self.concept_ret_label.setObjectName("search_widget")
		self.concept_ret_label.setStyleSheet("background-color:rgb(230, 230, 230);font-size:30px;color:rgb(50, 50, 50);"
										"border-style: solid;border-width: 1px;border-radius: 15px;border-color: rgb(50,50,50)")
		self.concept_ret_label.setAlignment(QtCore.Qt.AlignCenter)

		self.concept_ret_button = QtWidgets.QCheckBox(self.concept_ret_label)
		self.concept_ret_button.setGeometry(QtCore.QRect(1,1,method_name_width,radio_selection_y-2))
		self.concept_ret_button.setStyleSheet("padding-left:10px;border-style: none;")
		self.concept_ret_button.setText(_translate("MainWindow", "Caption Retrieval"))

		self.get_concept_button = QtWidgets.QPushButton(self.concept_ret_label)
		self.get_concept_button.setGeometry(QtCore.QRect((method_label_width/3 - button_width*1.5)/2,radio_selection_y+cl/5,
			button_width*1.5,button_height))
		self.get_concept_button.setStyleSheet("background-color:rgb(210, 210, 210);color:rgb(50, 50, 50);font-size:30px;border-style: outset;")
		self.get_concept_button.setText(_translate("MainWindow", "Get_Concepts"))

		self.concept_search_button = QtWidgets.QPushButton(self.concept_ret_label)
		self.concept_search_button.setGeometry(QtCore.QRect(cl/3,(radio_selection_y*2)+(cl*2)/5,
			button_width,button_height))
		self.concept_search_button.setStyleSheet("background-color:rgb(210, 210, 210);color:rgb(50, 50, 50);font-size:30px;border-style: outset;")
		self.concept_search_button.setText(_translate("MainWindow", "Search"))

		self.concept_search_number = QtWidgets.QLineEdit(self.concept_ret_label)
		self.concept_search_number.setGeometry(QtCore.QRect((cl*2)/3+button_width,(radio_selection_y*2)+(cl*2)/5,
				(method_label_width/3)-cl-button_width,button_height))
		self.concept_search_number.setStyleSheet("background-color:rgb(250, 250, 250);color:rgb(50, 50, 50)"
			";border-radius: 10px;font-size:30px;padding-left:10px")
		self.concept_search_number.setPlaceholderText("10")

		self.url_tb_label = QtWidgets.QLabel(self.query_selection)
		self.url_tb_label.setGeometry(QtCore.QRect(cl*0.5+button_width, 
							query_selection_height-(radio_selection_y*2)-button_height*2-(cl*1.25), 
							query_selection_width-(cl*0.5+button_width),radio_selection_y))
		self.url_tb_label.setObjectName("search_widget")
		self.url_tb_label.setStyleSheet("background-color:rgb(230, 230, 230);font-size:30px;color:rgb(50, 50, 50);"
										"border-style: solid;border-width: 0px;border-radius: 15px;border-color: rgb(50,50,50)")
		self.url_tb_label.setAlignment(QtCore.Qt.AlignCenter)

		self.url_tb = QtWidgets.QLineEdit(self.url_tb_label)
		self.url_tb.setGeometry(QtCore.QRect(0, 2,
				url_box_width,button_height))
		self.url_tb.setStyleSheet("background-color:rgb(250, 250, 250);color:rgb(50, 50, 50)"
			";border-radius: 10px;font-size:30px;padding-left:10px")
		self.url_tb.setPlaceholderText("Enter valid url for image download")

		similarity_distance_width = query_selection_width - (button_width + cl/2)

		self.similarity_distance_label = QtWidgets.QLabel(self.query_selection)
		self.similarity_distance_label.setGeometry(QtCore.QRect(query_selection_width-similarity_distance_width, 
							query_selection_height-(radio_selection_y*3)-button_height*2-(cl*1.5), 
							similarity_distance_width,radio_selection_y))
		self.similarity_distance_label.setObjectName("similarity_distance")
		self.similarity_distance_label.setStyleSheet("background-color:rgb(230, 230, 230);font-size:30px;color:rgb(50, 50, 50);"
										"border-style: solid;border-width: 1px;border-radius: 15px;border-color: rgb(50,50,50)")
		self.similarity_distance_label.setAlignment(QtCore.Qt.AlignCenter)

		similarity_button_width = similarity_distance_width/3

		self.mse_button = QtWidgets.QRadioButton(self.similarity_distance_label)
		self.mse_button.setGeometry(QtCore.QRect(1,1,similarity_button_width,radio_selection_y-2))
		self.mse_button.setStyleSheet("padding-left:10px;border-style: none;")
		self.mse_button.setText(_translate("MainWindow", "Jaccard Distance"))
		self.mse_button.setChecked(True)
		#self.mse_button.toggled.connect()

		self.euc_button = QtWidgets.QRadioButton(self.similarity_distance_label)
		self.euc_button.setGeometry(QtCore.QRect(similarity_button_width,1,similarity_button_width,radio_selection_y-2))
		self.euc_button.setStyleSheet("padding-left:10px;border-style: none;")
		self.euc_button.setText(_translate("MainWindow", "Euclidean distance"))

		self.cod_button = QtWidgets.QRadioButton(self.similarity_distance_label)
		self.cod_button.setGeometry(QtCore.QRect(similarity_button_width*2,1,similarity_button_width,radio_selection_y-2))
		self.cod_button.setStyleSheet("padding-left:10px;border-style: none;")
		self.cod_button.setText(_translate("MainWindow", "City Block distance"))

		self.search_number_c = QtWidgets.QPushButton(self.query_selection)
		self.search_number_c.setGeometry(QtCore.QRect(query_selection_width-button_width, query_image_height-button_height*2-cl/2,
				button_width,button_height))
		self.search_number_c.setStyleSheet("background-color:rgb(210, 210, 210);color:rgb(50, 50, 50);font-size:30px")
		self.search_number_c.setText(_translate("MainWindow", "Clear"))
		
	def retranslateUi(self, MainWindow):
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

	def check_cboxes(self):
		c = 0
		cboxes = [self.vgg_button,self.resnet_button,self.dense_button,self.ae_button]
		for cb in cboxes:
			if cb.isChecked() == True:
				c+=1
		if c >= 2:
			for cb in cboxes:
				if cb.isChecked() == False:
					cb.setEnabled(False)
		if c < 2:
			for cb in cboxes:
				if cb.isEnabled() == False:
					cb.setEnabled(True)
