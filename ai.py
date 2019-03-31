from sklearn.base import BaseEstimator
import math
import numpy
import sys
import warnings
import threading
import random


trainPercentage = 0.7

#	Program call:
#		python3 ai.py <optional output file>
#	The program will write out the error of the tests to the output file specified or to a default file in case no output is specified

#function theta[0] + theta[1]*x[0] + theta[2]*x[1] + ... + theta[n+1]*x[n]
def h (x, theta):
	#result = theta[0]
	result = theta[0] + sum ( [ x[j]*theta[j+1] for j in range(len(x)) ] )
	#for j in range (0, len(x)):
	#	result = result + x[j]*theta[j+1]
	return result

#derivative of the euclidean distance
def defaultCost (j, x, theta, y):
	#cost = 0
	if j != 0:
		cost = sum( [ (h(x[i], theta) - y[i])*x[i][j-1] for i in range(len(x)) ] )
		#for i in range(0, len(x)):
		#	with warnings.catch_warnings():
		#		warnings.filterwarnings('error')
		#		try:
		#			cost = cost + (y[i] - h(x[i], theta))*x[i][j-1]
		#		except Warning:
		#			print ("Theta values are too large and seem to be diverging")
		#			pass
	else:
		cost = sum( [ (h(x[i], theta) - y[i]) for i in range(len(x)) ] )
		#for i in range(0, len(x)):
		#	cost = cost + (h(x[i], theta) - y[i])

	return cost/len(x)

class CostRunner (threading.Thread):
	def __init__ (self, j, x, theta, y, costFunc=defaultCost):
		threading.Thread.__init__(self)
		self.j = j
		self.x = x
		self.theta = theta
		self.y = y
		self.costFunc = costFunc
		self.cost = float('inf')

	def run (self):
		self.cost = self.costFunc(self.j, self.x, self.theta, self.y)

class LMSTrainer(BaseEstimator):
	def __init__(self, analitic=False):

		self.analitic = analitic
		self.theta = None

	def fit(self, x, y, learningRate=0.000002, adaptiveLearning=False, learningAdaptionRate=0.1, costAdaptionStabilization = 0.0001, costFunc=defaultCost, convergenceThreshold=None, relativeThreshold=None, epochs=1):

		if relativeThreshold is None and convergenceThreshold is None:
			raise RuntimeError("Need to specify stop criteria")
		elif not (relativeThreshold is None or convergenceThreshold is None):
			raise RuntimeError("Can only use one stop criteria")

		print ("Training...")	#debug
		print ("Relative error: 0.0\nAbsolute error: 0.0\nConvergence rate: 0.0%")	#debug
		print ("Learning bias:", learningRate)	#debug
		if self.analitic:
			# TODO: FAZER POR MATRIZES
			pass
		else:
			random.seed()
			self.theta = [random.uniform(-10.0, 10.0) for i in range(0, len(x[0])+1)]	#first theta does not have associated x value

			relativeCost = float("inf")
			previousCost = float("inf")
			convergenceRate = float("inf")
			increasingRate = 1.0+learningAdaptionRate
			decreasingRate = 1.0-learningAdaptionRate
			it = 1
			while relativeThreshold is None and abs(100.0*convergenceRate) > convergenceThreshold or convergenceThreshold is None and relativeCost > relativeThreshold:

				for ep in range(epochs):
					evaluation = 0.0

					#prepare threads
					threads = []
					for j in range(0, len(self.theta)):
						threads.append(CostRunner(j, x, self.theta, y))

					#start threads
					for t in threads:
						t.start()

					#wait for threads to finish and evaluate the results
					evaluation = 0.0
					for t in threads:
						t.join()
						evaluation = evaluation + t.cost
					evaluation = evaluation/len(self.theta)

					if previousCost == float("inf"):
						relativeCost = abs(evaluation)
						previousCost = relativeCost
						lastConvergenceRate = 0.0
					else:
						if convergenceRate != float("inf"):
							lastConvergenceRate = convergenceRate
						convergenceRate = abs(previousCost)/abs(evaluation) - 1
						relativeCost = abs(evaluation - previousCost)
						previousCost = evaluation

					#debug
					print ("\033[4A\rRelative error:", relativeCost, "                   ")
					print ("Absolute error:", evaluation, "                ")
					print ("Convergence rate: {0:.2f}%   ".format(100.0*convergenceRate))
					print ("Learning bias:", learningRate, "                    ")

					#try to optimize learning bias
					if adaptiveLearning and it%(round(1.0/learningAdaptionRate)) == 0:

						if convergenceRate > 0.0:
							#attempt to find the highest possible value for the learning rate
							learningRate = learningRate*increasingRate
							increasingRate = (2.0-learningAdaptionRate)*(increasingRate**2.0 - 2.0*increasingRate + 2.0) - (1.0-learningAdaptionRate)	#reduce climbing speed the more you climb
							#increasingRate = increasingRate*math.exp(-increasingRate/relativeCost)

						else:
							#after climbing begin to reduce learning value to better approximate the result
							learningRate = learningRate*decreasingRate

							#reduce learning rate according to convergence rate. The bigger the improvements the more likely the next improvements will come from small steps
							decreasingRate = learningAdaptionRate*(-decreasingRate**2.0 + 2.0*decreasingRate) + (1.0 - learningAdaptionRate) #reduce decline speed the more you decline
							#decreasingRate = decreasingRate*math.exp(-decreasingRate/relativeCost)
							#if adaptionBias == 1.0:
							#	adaptionBias = 1.0 - learningAdaptionRate

					#update theta values
					for t in threads:
						self.theta[t.j] = self.theta[t.j] - learningRate*t.cost

				it = it+1



			# TODO: FAZERPELO GRADIENTE DESCENDETE
		print ("Training complete")	#debug

		return self

	def predict(self, x, y=None):

		if self.theta is None:
			raise RuntimeError("You must train classifer before predicting data!")

		prediction = self.theta[0]
		for j in range(1, len(self.theta)):
			prediction = prediction + self.theta[j]*x[j-1]

		return prediction

