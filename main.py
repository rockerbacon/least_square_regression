from ai import LMSTrainerBuilder, absolute_error_stop_function
from input import EnergyDataReader, RandomDataGenerator
from output import DatasetReaderObserver, TrainerObserver
import sys
import math

if len(sys.argv) == 1:
	output_file_path = "test_results.txt"
else:
	output_file_path = sys.argv[1]

#reader = EnergyDataReader()
reader = RandomDataGenerator(lambda x: x*5 - 2, 1, 20, -20, 20)

dataset_reader_observer = DatasetReaderObserver(sys.stdout)
reader.attach(dataset_reader_observer)

y, x = reader.read()

train_percentage = 0.7
dataset_slice_point = math.floor(train_percentage*len(x))
trainingset_x = x[:dataset_slice_point]
trainingset_y = y[:dataset_slice_point]
testingset_x = x[dataset_slice_point:]
testingset_y = y[dataset_slice_point:]

trainer_builder = LMSTrainerBuilder().with_defaults(trainingset_y, trainingset_x)
trainer_builder.with_stop_function(absolute_error_stop_function)

trainer = trainer_builder.build()

trainer_observer = TrainerObserver(sys.stdout)
trainer.attach(trainer_observer)

trainer.fit()

outputFile = open(output_file_path, "w")
print("\nTesting...")
print("Error: 0.0")
error = None
for i in range(0, len(testingset_x)):
	predictionError = abs(trainer.predict(testingset_x[i])-testingset_y[i])
	if error is None:
		error = predictionError
	else:
		error = (error+predictionError)/2.0
	outputFile.write(str(predictionError) + "\n")
	print("\033[1A\rError: {0:.6f}          ".format(error))
print("Finished testing")

outputFile.close()
