import csv
import numpy
import random
from exceptions import AbstractMethodCallError
from output import Observable

class DatasetReader(Observable):
	def __init__(self):
		super().__init__()
		self.__is_prepared_for_reading = False

	def prepare_for_reading (self, number_of_entries, number_of_variables):
		self.__y = numpy.array([0.0 for i in range(0, number_of_entries)])
		self.__x = numpy.array([[0.0 for j in range(0, number_of_variables)] for i in range(0, len(self.__y))])
		self.__current_row = 0
		self.__is_prepared_for_reading = True

	def total_data_size (self):
		return len(self.__y)

	def data_already_read (self):
		return self.__current_row

	def read_row (self, index, output_y, output_x):
		raise AbstractMethodCallError("read_row")

	def read(self):

		if not self.__is_prepared_for_reading:
			raise RuntimeError("Reader must be prepared by calling prepare_for_reading(number_of_entries, number_of_variables) before calling the method read()")

		self.notify_observers()

		number_of_rows = len(self.__y)
		while self.__current_row < number_of_rows:
			self.read_row(self.__current_row, self.__y, self.__x)
			self.__current_row += 1
			self.notify_observers()

		self.notify_observers()

		return self.__y, self.__x

class EnergyDataReader(DatasetReader):
	def __init__(self, input_file):
		super().__init__()

		self.__dataset = list(csv.reader(input_file, delimiter=","))

		self.__dataset = self.__dataset[1:] #skip header
		#random.shuffle(self.__dataset)

		self.__x_range_lower = 2
		self.__x_range_upper = 27
		self.__y_column_index = 1

		self.prepare_for_reading(len(self.__dataset), self.__x_range_upper-self.__x_range_lower)

	def read_row(self, index, output_y, output_x):

		output_y[index] = self.__dataset[index][self.__y_column_index]

		for i in range(self.__x_range_lower, self.__x_range_upper):
			if i != self.__y_column_index:
				output_x[index][i-self.__x_range_lower] = float(self.__dataset[index][i])

class RandomDataGenerator (DatasetReader):
	def __init__(self, function, number_of_function_variables, dataset_size, theta_lower_range=None, theta_upper_range=None):
		super().__init__()
		self.__function = function

		self.__theta_lower_range = -self.__size**2 if theta_lower_range is None else theta_lower_range
		self.__theta_upper_range = self.__size**2 if theta_upper_range is None else theta_upper_range

		random.seed()
		self.prepare_for_reading(dataset_size, number_of_function_variables)

	def read_row(self, index, output_y, output_x):
		for j in range (0, len(output_x[index])):
			output_x[index][j] = random.uniform(self.__theta_lower_range, self.__theta_upper_range)
		output_y[index] = self.__function(*output_x[index])

