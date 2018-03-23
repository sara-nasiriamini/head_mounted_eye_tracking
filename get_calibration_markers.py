import cv2

if __name__ == '__main__':

	img = cv2.imread("grid_new.jpg")
	crop_img = img[41:47, 15:21]
	cv2.imwrite("gird_1.png", crop_img)
	
	crop_img = img[45:51, 313:319]
	cv2.imwrite("gird_2.png", crop_img)
	
	crop_img = img[46:52, 605:611]
	cv2.imwrite("gird_3.png", crop_img)
	
	crop_img = img[137:143, 154:160]
	cv2.imwrite("gird_4.png", crop_img)
	
	crop_img = img[138:145, 458:464]
	cv2.imwrite("gird_5.png", crop_img)
	
	crop_img = img[242:248, 16:22]
	cv2.imwrite("gird_6.png", crop_img)
	
	crop_img = img[243:249, 311:317]
	cv2.imwrite("gird_7.png", crop_img)
	
	crop_img = img[244:250, 604:610]
	cv2.imwrite("gird_8.png", crop_img)
	
	crop_img = img[336:342, 153:159]
	cv2.imwrite("gird_9.png", crop_img)
	
	crop_img = img[337:343, 457:463]
	cv2.imwrite("gird_10.png", crop_img)
	
	crop_img = img[441:447, 15:21]
	cv2.imwrite("gird_11.png", crop_img)
	
	crop_img = img[442:448, 310:316]
	cv2.imwrite("gird_12.png", crop_img)
	
	crop_img = img[444:450, 605:611]
	cv2.imwrite("gird_13.png", crop_img)
	
	x_points = np.array([[18], [316], [608], [157], [461], [19], [314], [607], [156], [460], [18], [313], [608]])
	y_points = np.array([[44], [48], [49], [140], [141], [245], [246], [247], [339], [340], [444], [445], [447]])
	