import numpy as np

def find_mapping(points, vectors):
	A = np.array([1, points[0][0], points[0][1], points[0][0]*points[0][1], points[0][0]*points[0][0], points[0][1]*points[0][1], points[0][0]*points[0][0]*points[0][1]*points[0][1]])
	for i in xrange(1, points.shape[0]):
		row = np.array([1, points[i][0], points[i][1], points[i][0]*points[i][1], points[i][0]*points[i][0], points[i][1]*points[i][1], points[i][0]*points[i][0]*points[i][1]*points[i][1]])
		A = np.vstack((A, row))
	
	b = vectors
	
	x = np.linalg.lstsq(A,b,rcond=-1)[0]
	
	#x_1 = np.linalg.pinv(A)
	#x_1 = np.dot(x_1, b)
	
	return x