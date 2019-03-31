from ai import LMSTrainer
from input import EnergyDataReader, RandomDataGenerator, DatasetStatusObserver
import sys
import math

if len(sys.argv) == 1:
	output_file_path = "test_results.txt"
else:
	output_file_path = sys.argv[1]

#reader = EnergyDataReader()
reader = RandomDataGenerator(lambda theta0: theta0**2 + theta0*3 - 2, 1, 20, -20, 20)
reader.attach(DatasetStatusObserver(sys.stdout))

y, x = reader.read()

trainPercentage = 0.7
trainLimit = math.floor(trainPercentage*len(x))
trainX = x[:trainLimit]
trainY = y[:trainLimit]
testX = x[trainLimit:]
testY = y[trainLimit:]

trainer = LMSTrainer()

trainer.fit(trainX, trainY, adaptiveLearning=True, relativeThreshold=0.0000001)

outputFile = open(output_file_path, "w")
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
	print("\033[1A\rError: {0:.6f}          ".format(error))
print("Finished testing")

outputFile.close()
