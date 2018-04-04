from sklearn.base import BaseEstimator
import math
import csv
import numpy
import sys
import warnings
import threading
from copy import copy

trainPercentage = 0.7

#	Program call:
#		python3 ai.py <optional output file>
#	The program will write out the error of the tests to the output file specified or to a default file in case no output is specified

#function theta[0] + theta[1]*x[0] + theta[2]*x[1] + ... + theta[n+1]*x[n]
def h (x, theta):
	result = theta[0]
	for j in range (0, len(x)):
		result = result + x[j]*theta[j+1]
	return result

#derivative of the euclidean distance
def defaultCost (j, x, theta, y):
	cost = 0
	if j != 0:
		for i in range(0, len(x)):
			with warnings.catch_warnings():
				warnings.filterwarnings('error')
				try:
					cost = cost + (h(x[i], theta) - y[i])*x[i][j-1]
				except Warning:
					print ("Theta values are too large and seem to be diverging")
					pass
	else:
		for i in range(0, len(x)):
			cost = cost + (h(x[i], theta) - y[i])
			
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
		
#ignorar primeira linha (descricoes)
#ignorar primeira coluna (datas)
#ignorar segunda coluna (valor a ser estimado)
#ignorar duas ultimas colunas (valores dummy)
def openCsv (readFrom="energydata_complete.csv"):
		print("Reading dataset...\n0%")	#debug
		dataset = list(csv.reader(open(readFrom, "r"), delimiter=","))
		rows = len(dataset)
		colums = len(dataset[0])
		
		#extract correct values (y) from dataset
		y = []
		for i in range(1, rows):
			y.append(float(dataset[i][1]))
			progress = 100.0*i/(rows*(colums-3))	#debug
			print("\033[1A\r{0:.1f}%".format(progress) )	#debug
		
		#extract other values from dataset
		x = numpy.array([[0.0 for j in range(0, colums-4)] for i in range(0, rows-1)])
		i0 = 0
		for i1 in range(1, rows):
			j0 = 0
			for j1 in range(2, colums-2):
				x[i0][j0] = float(dataset[i1][j1])
				j0 = j0 + 1
			i0 = i0 + 1
			print("\033[1A\r{0:.1f}%".format( progress + 100.0*i1*(colums-4)/(rows*(colums-3)) ))	#debug

		print("Dataset ready")	#debug
		
		return x, y

class LMSTrainer(BaseEstimator):
	def __init__(self, analitic=False):
	
		self.analitic = analitic
		self.theta = None
 		
	def fit(self, x, y, learningRate=1.0, costFunc=defaultCost, threshold=None, iterations=None):
	
		if threshold is None and iterations is None:
			raise RuntimeError("Need to specify stop criteria")
		elif not (threshold is None or iterations is None):
			raise RuntimeError("Can only use one stop criteria")
			
		print ("Training...")	#debug
		if iterations is not None: print ("0.0%")	#debug
		print ("Relative error: 0.0\nConversion rate: 0.0%")	#debug
		if self.analitic:
			# TODO: FAZER POR MATRIZES
			pass
		else:
			self.theta = [1.0 for i in range(0, len(x[0])+1)]	#first theta does not have associated x value
			
			relativeCost = float("inf")
			previousCost = None
			it = 0
			while threshold is None and it < iterations or iterations is None and relativeCost > threshold:
				evaluation = 0.0
				
				#prepare threads
				threads = []
				for j in range(0, len(self.theta)):
					threads.append(CostRunner(j, x, self.theta, y))
				
				#start threads
				for t in threads:
					t.start()
				
				#wait for threads to finish
				for t in threads:
					t.join()
					
				#update theta values and calculate costs
				evaluation = 0.0
				for t in threads:
					self.theta[t.j] = self.theta[t.j] - learningRate*t.cost
					evaluation = evaluation + t.cost
					
				#debug
				#if iterations is not None:
				#	print ("\033[2A\r{0:.1f}%".format( 100.0*(it*len(self.theta) + j + 1)/(iterations*len(self.theta)) ))
				
				evaluation = evaluation/len(self.theta)		
				if previousCost is None:
					relativeCost = abs(evaluation)
					previousCost = relativeCost
					conversionRate = 0.0
				else:
					conversionRate = 100.0*(relativeCost/abs(evaluation - previousCost) - 1)
					relativeCost = abs(evaluation - previousCost)
					previousCost = evaluation
				
				print ("\033[2A\rRelative error: {0:.5f}            ".format(relativeCost))
				print ("Conversion rate: {0:.1f}%   ".format(conversionRate))
						
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

x, y = openCsv()
trainLimit = math.floor(trainPercentage*len(x))

trainer = LMSTrainer()
if len(sys.argv) == 1:
	outputFile = open("test_results.txt", "w")
else:
	outputFile = open(sys.argv[1], "w")

trainX = x[:trainLimit]
trainY = y[:trainLimit]
#print (trainY)	#debug
trainer.fit(trainX, trainY, learningRate=0.000002, threshold=0.0001)

testX = x[trainLimit:]
testY = y[trainLimit:]
print("Testing...")
print("Error: 0.0")
error = None
for i in range(0, len(testX)):
	predictionError = abs(trainer.predict(testX[i])-testY[i])
	if error is None:
		error = predictionError
	else:
		error = (error+predictionError)/2.0
	outputFile.write(str(predictionError) + "\n")
	print("\033[1A\rError: {0:.6f}".format(error))
print("Finished testing")
	
outputFile.close()

