import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
import time
import argparse
import ast
import matplotlib.pyplot as plt
import argparse

from api import PRN
from utils.render import render_texture
import cv2
import dlib
from demo_texture import texture_edit
import tensorflow as tf

def detectFace(image, prn, mode) : 
	[h, w, c] = image.shape
	cnn_face_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	output = image

	# apply face detection (cnn)
	detected_faces = cnn_face_detector(rgb, 1)

	if mode == 2 :
		
		d1 = detected_faces[0].rect 
		left1 = d1.left(); right1 = d1.right(); top1 = d1.top(); bottom1 = d1.bottom()

		d2 = detected_faces[1].rect 
		left2 = d2.left(); right2 = d2.right(); top2 = d2.top(); bottom2 = d2.bottom()

		image1 = image[top1-20:bottom1+20, left1-20:right1+20]
		image2 = image[top2-20:bottom2+20, left2-20:right2+20]

		output1 = texture_edit(prn, image1, image2)
		output2 = texture_edit(prn, image2, image1)

		image[top1-20:bottom1+20, left1-20:right1+20] = output1
		image[top2-20:bottom2+20, left2-20:right2+20] = output2

		output = image

	elif mode == 1: 
		image2 = cv2.imread('joke.jpg')
		output = texture_edit(prn, image, image2)
		output = image

	else :
		print('Wrong Mode selected')

	return output

if __name__ == "__main__" :
	start = time.time()
	image = cv2.imread('scarletts0.jpg')
	# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
	os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
	# tf.debugging.set_log_device_placement(True)
	# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
	# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	prn = PRN(is_dlib = True)
	img = detectFace(image, prn, 1)
	cv2.imwrite('Face.jpg', img)
	end = time.time()
	print(f'Time taken : {end-start}')