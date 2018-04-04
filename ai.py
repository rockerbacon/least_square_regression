from sklearn.base import BaseEstimator
import math
import csv
import numpy
import sys

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
	if j ~= 0:
		for i in range(0, len(x)):
			cost = cost + (h(x[i], theta) - y[i])*x[i][j]
	else:
		for i in range(0, len(x)):
			cost = cost + (h(x[i], theta) - y[i])
			
	return cost/len(x)
		
#ignorar primeira linha (descricoes)
#ignorar primeira coluna (datas)
#ignorar segunda coluna (valor a ser estimado)
#ignorar duas ultimas colunas (valores dummy)
def openCsv (readFrom="energydata_complete.csv"):
		print("Reading dataset...")	#debug
		dataset = list(csv.reader(open(readFrom, "r"), delimiter=","))
		rows = len(dataset)
		colums = len(dataset[0])
		
		#extract correct values (y) from dataset
		y = []
		for i in range(1, rows):
			y.append(float(dataset[i][1]))
		
		#extract other values from dataset
		x = numpy.array([[0.0 for j in range(0, colums-4)] for i in range(0, rows-1)])
		i0 = 0
		for i1 in range(1, rows):
			j0 = 0
			for j1 in range(2, colums-2):
				x[i0][j0] = float(dataset[i1][j1])
				j0 = j0 + 1
			i0 = i0 + 1

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
			
		print "Training...\n0%"	#debug
		if self.analitic:
			# TODO: FAZER POR MATRIZES
			pass
		else:
			self.theta = [1.0 for i in range(0, len(x[0])+1)]	#first theta does not have associated x value
			
			relativeCost = float("inf")
			firstCost = None
			it = 0
			while threshold is None and it < iterations or iterations is None and relativeCost > threshold:
				evaluation = 0.0
				for j in range(0, self.theta):
					cost = costFunc(j, x, self.theta, y)
					evaluation = evaluation + cost
					self.theta[j] = self.theta[j] - learningRate*cost
				it = it + 1
				
				evaluation = evaluation/len(self.theta)
				
				if firstCost is None:
					relativeCost = math.abs(evaluation)
					firstCost = relativeCost
				else:
					relativeCost = math.abs(relativeCost - evaluation)
				
				#debug
				if iterations is not None:
					print "\033[1A\r"+str(100.0*it/iterations)+"%"
				elif threshold is not None:
					print "\033[1A\r" + str( 100.0*(firstCost + threshold - relativeCost) / firstCost ) + "%"
			
			# TODO: FAZERPELO GRADIENTE DESCENDETE
		print "Training complete"	#debug
		
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
if sys.argv[1] is None:
	outputFile = open("test_results.txt", "w")
else:
	outputFile = open(sys.argv[1], "w")

trainX = x[:trainLimit]
trainY = y[:trainLimit]
trainer.fit(trainX, trainY, iterations=100)

testX = x[trainLimit:]
testY = y[trainLimit:]
for i in range(0, testX):
	outputFile.write(str(math.abs(trainer.predict(x)-testY[i])) + "\n")
	
outputFile.close()

