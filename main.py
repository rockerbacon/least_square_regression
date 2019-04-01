from ai import LMSTrainerBuilder, absolute_error_stop_function, minibatch_error_evaluation_function, batch_error_evaluation_function
from input import EnergyDataReader, RandomDataGenerator
from output import DatasetReaderObserver, TrainerObserver
import sys
import math

if len(sys.argv) == 1:
	output_file_path = "test_results.txt"
else:
	output_file_path = sys.argv[1]

input_file = open("energydata_complete.csv", "r")
reader = EnergyDataReader(input_file)
#reader = RandomDataGenerator(lambda x0, x1, x2: x0*2 + x1*5 + x2*3 - 2, 3, 200, -20, 20)

dataset_reader_observer = DatasetReaderObserver(sys.stdout)
reader.attach(dataset_reader_observer)

y, x = reader.read()

input_file.close()

train_percentage = 0.7
dataset_slice_point = math.floor(train_percentage*len(x))
trainingset_x = x[:dataset_slice_point]
trainingset_y = y[:dataset_slice_point]
testingset_x = x[dataset_slice_point:]
testingset_y = y[dataset_slice_point:]

trainer_builder = LMSTrainerBuilder().with_defaults(trainingset_y, trainingset_x)

trainer_builder.with_error_evaluation_function(lambda y, x, theta, prediction_function: minibatch_error_evaluation_function(y, x, theta, prediction_function, 0.05))
trainer_builder.with_learning_bias(0.00000000002)
trainer_builder.with_initial_theta_values([0 for i in range(len(x[0])+1)])

# trainer_builder.with_error_evaluation_function(batch_error_evaluation_function)
# trainer_builder.with_learning_bias(0.000085)

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
