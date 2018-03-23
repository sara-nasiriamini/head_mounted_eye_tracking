import cv2

if __name__ == '__main__':
	for i in xrange(0,13):
		if (i == 0):
			img = cv2.imread("cap1.jpeg")
			crop_img = img[90:340, 110:360]
			cv2.imwrite("c_1_crop.bmp", crop_img)

		if (i == 1):
			img = cv2.imread("cap2.jpeg")
			crop_img = img[90:340, 110:360]
			cv2.imwrite("c_2_crop.bmp", crop_img)
		
		if (i == 2):
			img = cv2.imread("cap3.jpeg")
			crop_img = img[90:340, 110:360]
			cv2.imwrite("c_3_crop.bmp", crop_img)
		if (i == 3):
			img = cv2.imread("cap4.jpeg")
			crop_img = img[90:340, 110:360]
			cv2.imwrite("c_4_crop.bmp", crop_img)
		
		if (i == 4):
			img = cv2.imread("cap5.jpeg")
			crop_img = img[90:340, 110:360]
			cv2.imwrite("c_5_crop.bmp", crop_img)
			
		if (i == 5):
			img = cv2.imread("cap6.jpeg")
			crop_img = img[90:340, 110:360]
			cv2.imwrite("c_6_crop.bmp", crop_img)

		if (i == 6):
			img = cv2.imread("cap7.jpeg")
			crop_img = img[90:340, 110:360]
			cv2.imwrite("c_7_crop.bmp", crop_img)
		
		if (i == 7):
			img = cv2.imread("cap8.jpeg")
			crop_img = img[90:340, 110:360]
			cv2.imwrite("c_8_crop.bmp", crop_img)
		if (i == 8):
			img = cv2.imread("cap9.jpeg")
			crop_img = img[90:340, 110:360]
			cv2.imwrite("c_9_crop.bmp", crop_img)
		
		if (i == 9):
			img = cv2.imread("cap10.jpeg")
			crop_img = img[90:340, 110:360]
			cv2.imwrite("c_10_crop.bmp", crop_img)
			
		if (i == 10):
			img = cv2.imread("cap11.jpeg")
			crop_img = img[90:340, 110:360]
			cv2.imwrite("c_11_crop.bmp", crop_img)

		if (i == 11):
			img = cv2.imread("cap12.jpeg")
			crop_img = img[90:340, 110:360]
			cv2.imwrite("c_12_crop.bmp", crop_img)
		
		if (i == 12):
			img = cv2.imread("cap13.jpeg")
			crop_img = img[90:340, 110:360]
			cv2.imwrite("c_13_crop.bmp", crop_img)