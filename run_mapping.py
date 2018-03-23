import numpy as np
import cv2
from scipy.optimize import curve_fit
from fit_glint_circle import *
from pupil_detection import *
from map_displacement_vector import *

def find_glint_pupil(img, img_d, name):
	# convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	(px,py) = find_pupil(img_d, gray, name)
	pupil_center = (px,py)
	
	p_c_x = px - 60
	p_c_y = py - 40
	p_c_x_2 = px - 60 + 110
	p_c_y_2 = py - 40 + 95
	
	# convert to grayscale
	gray = cv2.cvtColor(img_d, cv2.COLOR_BGR2GRAY)
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
	new_cont = []
	new_c = []
	centers_x = []
	centers_y = []
	for x in contours:
		area = cv2.contourArea(x)
		if (area < 100):
			M = cv2.moments(x)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			if (p_c_x < cx < p_c_x_2 and p_c_y < cy < p_c_y_2):
				new_c.append(x)
				centers_x.append(cx)
				centers_y.append(cy)
				new_cont.append(x)
				
	xc_2, yc_2, R_2 = find_circle(np.asarray(centers_x), np.asarray(centers_y))
	glint_center = (xc_2, yc_2)
	
	
	img_c = cv2.imread(name)
	cv2.drawContours(img_c, new_cont, -1, (0,255,0), 3)
	
	center_rec = (px - 60, py - 40)
	center_rec_2 = (px - 60 + 110, py - 40 + 95)
	
	cv2.circle(img_c, pupil_center, 4, (255,0,255), 2)
	
	cv2.circle(img_c,(int(xc_2), int(yc_2)), 4, (0,255,255), 2)
	cv2.circle(img_c,(int(xc_2), int(yc_2)), int(R_2), (255,255,0), 1)
	cv2.imwrite(name,img_c)
	
	vector = np.asarray(pupil_center) - np.asarray(glint_center)
	
	return vector, glint_center, pupil_center
	
if __name__ == '__main__':
	grid = cv2.imread('grid_new.jpg')

	
	x_points = np.array([[18], [316], [608], [314], [18], [313], [608]])
	y_points = np.array([[44], [48], [49], [246], [444], [445], [447]])
	
	
	vectors = []
	glint_centers = []
	pupil_centers = []

	name = 'test.png'
	for i in xrange(0, 13):
		if (i == 0):
			img = cv2.imread('masked_c_1.png')
			img_d = cv2.imread('c_1_crop.bmp')
			name = 'c_1_center.png'
		
		if (i == 1):
			img = cv2.imread('masked_c_2.png')
			img_d = cv2.imread('c_2_crop.bmp')
			name = 'c_2_center.png'
		
		if (i == 2):
			img = cv2.imread('masked_c_3.png')
			img_d = cv2.imread('c_3_crop.bmp')
			name = 'c_3_center.png'
			
		if (i == 3):
			img = cv2.imread('masked_c_4.png')
			img_d = cv2.imread('c_4_crop.bmp')
			name = 'c_4_center.png'
		
		if (i == 4):
			img = cv2.imread('masked_c_5.png')
			img_d = cv2.imread('c_5_crop.bmp')
			name = 'c_5_center.png'

		if (i == 5):
			img = cv2.imread('masked_c_6.png')
			img_d = cv2.imread('c_6_crop.bmp')
			name = 'c_6_center.png'
			
		if (i == 6):
			img = cv2.imread('masked_c_7.png')
			img_d = cv2.imread('c_7_crop.bmp')
			name = 'c_7_center.png'
		
		if (i == 7):
			img = cv2.imread('masked_c_8.png')
			img_d = cv2.imread('c_8_crop.bmp')
			name = 'c_8_center.png'
		
		if (i == 8):
			img = cv2.imread('masked_c_9.png')
			img_d = cv2.imread('c_9_crop.bmp')
			name = 'c_9_center.png'
			
		if (i == 9):
			img = cv2.imread('masked_c_10.png')
			img_d = cv2.imread('c_10_crop.bmp')
			name = 'c_10_center.png'
		
		if (i == 10):
			img = cv2.imread('masked_c_11.png')
			img_d = cv2.imread('c_11_crop.bmp')
			name = 'c_11_center.png'
			
		if (i == 11):
			img = cv2.imread('masked_c_12.png')
			img_d = cv2.imread('c_12_crop.bmp')
			name = 'c_12_center.png'
		
		if (i == 12):
			img = cv2.imread('masked_c_13.png')
			img_d = cv2.imread('c_13_crop.bmp')
			name = 'c_13_center.png'
		
		
		vector, glint_center, pupil_center = find_glint_pupil(img, img_d, name)
		if (i == 0 or i == 1 or i == 2 or i == 6 or i == 10 or i == 11 or i == 12):
			vectors.append(vector)
			glint_centers.append(glint_center)
			pupil_centers.append(pupil_center)

		
	g_c_mean = np.mean(np.array(glint_centers))
	vectors_new = np.array(pupil_centers) - g_c_mean
	
	a = find_mapping(vectors_new, x_points)
	b = find_mapping(vectors_new, y_points)
	
	for i in xrange(0,13):
		if (i == 0):
			img = cv2.imread('masked_c_1.png')
			img_d = cv2.imread('c_1_crop.bmp')
		
		if (i == 1):
			img = cv2.imread('masked_c_2.png')
			img_d = cv2.imread('c_2_crop.bmp')
		
		if (i == 2):
			img = cv2.imread('masked_c_3.png')
			img_d = cv2.imread('c_3_crop.bmp')
			
		if (i == 3):
			img = cv2.imread('masked_c_4.png')
			img_d = cv2.imread('c_4_crop.bmp')
		
		if (i == 4):
			img = cv2.imread('masked_c_5.png')
			img_d = cv2.imread('c_5_crop.bmp')	
		
		if (i == 5):
			img = cv2.imread('masked_c_6.png')
			img_d = cv2.imread('c_6_crop.bmp')
		
		if (i == 6):
			img = cv2.imread('masked_c_7.png')
			img_d = cv2.imread('c_7_crop.bmp')
		
		if (i == 7):
			img = cv2.imread('masked_c_8.png')
			img_d = cv2.imread('c_8_crop.bmp')
			
		if (i == 8):
			img = cv2.imread('masked_c_9.png')
			img_d = cv2.imread('c_9_crop.bmp')
					
		if (i == 9):
			img = cv2.imread('masked_c_10.png')
			img_d = cv2.imread('c_10_crop.bmp')
			
		if (i == 10):
			img = cv2.imread('masked_c_11.png')
			img_d = cv2.imread('c_11_crop.bmp')
		
		if (i == 11):
			img = cv2.imread('masked_c_12.png')
			img_d = cv2.imread('c_12_crop.bmp')
			
		if (i == 12):
			img = cv2.imread('masked_c_13.png')
			img_d = cv2.imread('c_13_crop.bmp')
		
			
		name = 'test.png'
		vector, glint_center, pupil_center = find_glint_pupil(img, img_d, name)
		
		(u,v) = np.array(pupil_center) - g_c_mean
		a_pg = np.array([1, u, v, u*v, u*u, v*v, u*u*v*v])
		x_c = np.dot(a_pg, a)
		y_c = np.dot(a_pg, b)
	
		if (i == 0 or i == 1 or i == 2 or i == 6 or i == 10 or i == 11 or i == 12):
			cv2.circle(grid,(int(x_c[0]), int(y_c[0])),6,(255,0,0),2)
		if (i == 3 or i == 4 or i == 5 or i == 7 or i == 8 or i == 9):
			cv2.circle(grid,(int(x_c[0]), int(y_c[0])),6,(255,0,255),2)
		cv2.imwrite("grid_tested.png", grid)