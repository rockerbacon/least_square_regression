import random

def function (x):
	return 2*x + 3
	
def random_generate (size, lowerRange=None, upperRange=None):

	if lowerRange is None: lowerRange = -size**2.0
	if upperRange is None: upperRange = size**2.0

	random.seed()
	y = [0.0 for i in range(0, size)]
	x = [[random.uniform(lowerRange, upperRange)] for i in range(0, size)]
	
	for i in range(0, size):
		y[i] = function(x[i][0])
		
	return x, y
