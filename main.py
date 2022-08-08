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
from test3 import * 

def change_brightness(img, value):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	v = cv2.add(v,value)
	v[v > 255] = 255
	v[v < 0] = 0
	final_hsv = cv2.merge((h, s, v))
	img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	return img
	

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
	for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

cap = cv2.VideoCapture('abhi3.mp4')
vid = cv2.VideoWriter('faceabi2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920,1080))

prn = PRN(is_dlib = True)
start = time.time()
i = 0
ret = True
try : 
	while(ret):
		try : 
			ret, frame = cap.read()
			img = detectFace(frame, prn, mode)
			if i == 0 :
				cv2.imwrite(f'scarletts{i}.jpg', img)
				
			print(f'Running for frame : {i+1}')
			i += 1
			vid.write(img)
		except Exception as e :
			print(f'Exception occured neglecting frame : {e}')
			pass
  
except : 
	print('Exception occured neglecting code')
	pass

finally:
	print("Finally")

cap.release()
cv2.destroyAllWindows()
vid.release()

end = time.time()
print(f'Time taken : {end-start}')