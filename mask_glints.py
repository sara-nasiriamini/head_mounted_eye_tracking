import numpy as np
import cv2
from scipy.optimize import curve_fit
from fitting import *
from pupil_detection import *
from mapping import *

def detect_glint(img):
	# convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# blur 
	blur = cv2.GaussianBlur(gray,(3,3),0)
	# binarization
	val, bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# morph open and close
	st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	open = cv2.morphologyEx(bin, cv2.MORPH_OPEN, st)
	st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, st)
	# get contours
	im_contour, contours, hierarchy = cv2.findContours(close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# get contours of interest location and interest size
	new_c = []
	centers_x = []
	centers_y = []
	for x in contours:
		# get contours of interest size in intereset region for masking
		area = cv2.contourArea(x)
		if (area < 400):
			M = cv2.moments(x)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			if (30 < cx < 240 and 30 < cy < 200):
				new_c.append(x)
				centers_x.append(cx)
				centers_y.append(cy)

	
	# mask contours
	lst_pixels = []
	for i in range(len(new_c)):
		# Create a mask image that contains the contour filled in
		mask = np.zeros_like(gray)
		cv2.drawContours(mask, new_c, i, color=255, thickness=-1)
		# Access the image pixels and create a 1D numpy array then add to list
		pts = np.where(mask == 255)
		lst_pixels.append(pts)
	
	# mask with average value
	for i in xrange(0, np.asarray(lst_pixels).shape[0]):
		x = np.asarray(lst_pixels)[i][0]
		y = np.asarray(lst_pixels)[i][1]
		for idx in xrange(0, x.shape[0]):
			value = gray[x[idx] - 1][y[idx] - 1]
			gray[x[idx]][y[idx]] = value
	
	
	return gray
	

if __name__ == '__main__':
	
	for i in xrange(0,13):
		if (i == 0):
			img = cv2.imread('c_1_crop.bmp')
			img_r = detect_glint(img)
			cv2.imwrite("masked_c_1.png",img_r)
	
		if (i == 1):
			img = cv2.imread('c_2_crop.bmp')
			img_r = detect_glint(img)
			cv2.imwrite("masked_c_2.png",img_r)
		
		if (i == 2):
			img = cv2.imread('c_3_crop.bmp')
			img_r = detect_glint(img)
			cv2.imwrite("masked_c_3.png",img_r)
			
		if (i == 3):
			img = cv2.imread('c_4_crop.bmp')
			img_r = detect_glint(img)
			cv2.imwrite("masked_c_4.png",img_r)
		
		if (i == 4):
			img = cv2.imread('c_5_crop.bmp')
			img_r = detect_glint(img)
			cv2.imwrite("masked_c_5.png",img_r)
		
		if (i == 5):
			img = cv2.imread('c_6_crop.bmp')
			img_r = detect_glint(img)
			cv2.imwrite("masked_c_6.png",img_r)
			
		if (i == 6):
			img = cv2.imread('c_7_crop.bmp')
			img_r = detect_glint(img)
			cv2.imwrite("masked_c_7.png",img_r)
		
		if (i == 7):
			img = cv2.imread('c_8_crop.bmp')
			img_r = detect_glint(img)
			cv2.imwrite("masked_c_8.png",img_r)
		
		if (i == 8):
			img = cv2.imread('c_9_crop.bmp')
			img_r = detect_glint(img)
			cv2.imwrite("masked_c_9.png",img_r)
			
		if (i == 9):
			img = cv2.imread('c_10_crop.bmp')
			img_r = detect_glint(img)
			cv2.imwrite("masked_c_10.png",img_r)
		
		if (i == 10):
			img = cv2.imread('c_11_crop.bmp')
			img_r = detect_glint(img)
			cv2.imwrite("masked_c_11.png",img_r)
			
		if (i == 11):
			img = cv2.imread('c_12_crop.bmp')
			img_r = detect_glint(img)
			cv2.imwrite("masked_c_12.png",img_r)
		
		if (i == 12):
			img = cv2.imread('c_13_crop.bmp')
			img_r = detect_glint(img)
			cv2.imwrite("masked_c_13.png",img_r)
	