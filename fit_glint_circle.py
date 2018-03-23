from numpy import *
import functools
from scipy import optimize


def find_circle(x, y):
	# coordinates of estimated center
	x_m = mean(x)
	y_m = mean(y)

	# count functions calls
	def count(function):

		@functools.wraps(function)
		def wrapped(*args):
			wrapped.ncalls +=1
			return function(*args)

		wrapped.ncalls = 0
		return wrapped

	def calc_radius(x_c, y_c):
		# calculate the distance of each 2D points from the center
		return sqrt((x-x_c)**2 + (y-y_c)**2)

	@count
	def find_2(c):
		# calculate the distance between the 2D points and the mean circle centered
		r_i = calc_radius(*c)
		return r_i - r_i.mean()

	center_estimate = x_m, y_m
	center_2, ier = optimize.leastsq(find_2, center_estimate)

	x_c_2, y_c_2 = center_2
	r_i_2       = calc_radius(x_c_2, y_c_2)
	r_2        = r_i_2.mean()
	residu_2   = sum((r_i_2 - r_2)**2)
	residu2_2  = sum((r_i_2**2-r_2**2)**2)
	ncalls_2   = find_2.ncalls

	return x_c_2, y_c_2, r_2