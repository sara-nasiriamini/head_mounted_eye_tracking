import cv2
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def find_pupil(img_d, img, name):
	img_copy = img
	gray = img
	
	# binary threshold
	val, bin = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
	st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	open = cv2.morphologyEx(bin, cv2.MORPH_OPEN, st)
	st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, st)
	img = close
	eye_rec = (0,0,img.shape[1],img.shape[0])
	
	# resize the image
	scaled_w = 50
	eye_roi_scaled = cv2.resize(img, (scaled_w, (scaled_w*img.shape[0])/img.shape[1]))
	
	# calculate gradient
	gradient_x = compute_gradient(eye_roi_scaled);
	gradient_y = (compute_gradient(eye_roi_scaled.T)).T;
	
	mag = matrix_magnitude(gradient_x, gradient_y)
	
	# get threshold value for gradient filtering
	threshold = dynamic_threshold(mag)
	
	for i in xrange(0, mag.shape[0]):
		mag_r = mag[i]
		for j in xrange(0, mag.shape[1]):
			x_p = gradient_x[i][j]
			y_p = gradient_y[i][j]
			mag_p = mag[i][j]
			if (mag_p > threshold):
				gradient_x[i][j] = gradient_x[i][j]/mag_p
				gradient_y[i][j] = gradient_y[i][j]/mag_p
			else: 
				gradient_x[i][j] = 0.0
				gradient_y[i][j] = 0.0

	blur = eye_roi_scaled
	for i in xrange(0, blur.shape[0]):
		for j in xrange(0, blur.shape[1]):
			blur[i][j] = 255 - blur[i][j]
	
	out = np.zeros((blur.shape[0], blur.shape[1]), dtype='float')
	for i in xrange(0, blur.shape[0], 1):
		for j in xrange(0, blur.shape[1], 1):
			if (gradient_x[i][j] == 0.0 and gradient_y[i][j] == 0.0):
				continue
			possible_centers(j, i, blur, gradient_x[i][j], gradient_y[i][j], out);
		
	num_gradients = blur.shape[0]*blur.shape[1]
	
	res = np.float32(out)
	res = res*(1/num_gradients)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(out)
	
	center = unscale_point(max_loc, eye_rec)
	(x,y) = center
	
	center = (int(x), int(y))
	
	(x,y) = center 
	center_rec = (int(x) - 100, int(y) - 70)
	center_rec_2 = (int(x) - 100 + 200, int(y) - 70 + 170)
	cv2.imwrite(name, img_d)
	return center

def compute_gradient(roi_eye):
	output = np.empty((roi_eye.shape[0],roi_eye.shape[1]), dtype='float')
	
	for i in xrange(0, output.shape[0]):
		
		output[i][0] = int(roi_eye[i][1]) - int(roi_eye[i][0])
		
		for j in xrange(1, output.shape[1] - 1):
			output[i][j] = (int(roi_eye[i][j+1]) - int(roi_eye[i][j-1]))/2
		
		output[i][output.shape[1] - 1] = int(roi_eye[i][output.shape[1] - 1]) - int(roi_eye[i][output.shape[1] - 2])		
	
	return output

def matrix_magnitude(gradient_x, gradient_y): 
	output = np.empty((gradient_x.shape[0],gradient_x.shape[1]), dtype='float')
	
	for i in xrange(0, gradient_x.shape[0]):
		raw_x = gradient_x[i]
		raw_y = gradient_y[i]
		for j in xrange(0, gradient_x.shape[1]):
			g_x = raw_x[j]
			g_y = raw_y[j]
			output[i][j] = np.sqrt((g_x*g_x) + (g_y*g_y))

	return output
	
def dynamic_threshold(magnitude):
	gradient_threshold = 5.0
	std = np.std(magnitude)
	std = std/np.sqrt(magnitude.shape[0]*magnitude.shape[1])
	mean = np.mean(magnitude)
	threshold = gradient_threshold*std + mean
	return threshold
	
def possible_centers(x, y, invert, gx, gy, center):
	for i in xrange(0, invert.shape[0], 1):
		for j in xrange(0, invert.shape[1], 1):
			if (x == j and y == i):
				continue
			dx = x - j;
			dy = y - i;
			magnitude = np.sqrt((dx * dx) + (dy * dy));
			dx = dx / magnitude;
			dy = dy / magnitude;
			dotProduct = dx*gx + dy*gy;
			dotProduct = np.maximum(0.0, dotProduct)
			center[i][j] = center[i][j] + (dotProduct * dotProduct)
	
	return
	
def unscale_point(point, orig_size):
	(x,y,w,h) = orig_size
	ratio = float(50/w)
	(x,y) = point
	px = round(x*w/50)
	py = round(y*w/50)
	return (px,py)
	
if __name__ == '__main__':
	# get image
	img = cv2.imread('masked_c_1.png')
	img_d = cv2.imread('c_1_crop.bmp')
	# convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	find_pupil(img_d, gray, 'test.png')