# imports
from keras.models import load_model
import cv2
import numpy as np
import os
import os.path
import time

# load model
ld_model = load_model('keras_LeNet.h5')


while(True):
	if(os.path.exists('test.jpg')):
	
		# load image
		image = cv2.imread('test.jpg')

		# convert to gray scale
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# resize
		input_img = cv2.resize(gray_image, (28, 28))

		# reshape
		input_img_r = input_img[np.newaxis, np.newaxis, :, :]

		# predict
		pred = ld_model.predict(input_img_r)

		prediction = pred.argmax(axis=1)

		os.system('say ' + str(prediction[0]))

		os.remove('test.jpg')
	
	time.sleep(5)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
