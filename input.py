import csv
import numpy
import random

class Reader():
	def __init__(self):
		self.__observers = []

	def attach (self, observer):
		self.__observers.append(observer)

	def notify_observers (self):
		for observer in self.__observers: observer.update(self)

	def read(self):
		return None

	def total_data_size (self):
		return 1.0

	def data_read (self):
		return 0.0


class DatasetStatusObserver():
	def __init__(self, output):
		self.reset()
		if (not output.writable()):
			raise RuntimeError("Need writable output to create observer")
		self.__output = output

	def reset(self):
		self.__reading_has_begun = False

	def update(self, reader):
		progress = reader.data_read()/reader.total_data_size()*100

		if (not self.__reading_has_begun):
			self.__reading_has_begun = True
			self.__output.write("\nReading dataset...\n0%")

		output.write("\033[1A\r{0:.1f}%\n".format(progress))

		if progress >= 100:
			output.write("Dataset read successfuly\n")
			self.reset()


class EnergyDataReader(Reader):
	def __init__(self, input_file_path="energydata_complete.csv"):
		super().__init__()
		self.__input_file_path = open(input_file_path, "r")

	def total_data_size(self):
		return len(self.__dataset)

	def data_read(self):
		return self.__current_row

	def __open_csv_for_reading(self):

		input_file = open(self.__input_file_path, "r")
		self.__dataset = list(csv.reader(input_file, delimiter=","))
		input_file.close()

		self.__dataset = self.__dataset[1:] #skip header
		random.shuffle(self.__dataset)

		self.__current_row = 0

	def __read_row(self, output_y, output_x):

		if self.__has_something_to_read():

			output_y[self.__current_row] = self.__dataset[self.__current_row][self.__y_column_index]

			for i in range(self.__x_range_lower, self.__x_range_upper):
				if i != self.__y_column_index:
					output_x[self.__current_row][i-self.__x_range_lower] = float(self.__dataset[self.__current_row][i])

			self.__current_row += 1

	def __set_y_column(self, column_index):
		self.__y_column_index = column_index

	#range [column_range_lower, column_range_upper)
	def __set_x_column_range (column_range_lower, column_range_upper):
		self.__x_range_lower = column_range_lower
		self.__x_range_upper = column_range_upper

	def __has_something_to_read (self):
		return self.__current_row < len(self.__dataset)

	def read(self):

		self.notify_observers()

		self.__open_csv_for_reading()
		self.__set_y_column(1)
		self.__set_x_column_range(2, 27)

		y = numpy.array([0.0 for i in range(0, self.total_data_size())])
		x = numpy.array([[0.0 for j in range(0, self.__x_range_upper-self.__x_range_lower)] for i in range(0, len(y))])
		while self.__has_something_to_read():
			self.__read_row(y, x)
			self.notify_observers()

		self.notify_observers()

		return y, x

class RandomDataGenerator (Reader):
	def __init__(self, function, number_of_function_variables, dataset_size, theta_lower_range=None, theta_upper_range=None):
		super().__init__()
		self.__function = function
		self.__number_of_function_variables = number_of_function_variables
		self.__size = dataset_size
		self.__amount_generated = 0

		self.__theta_lower_range = -self.__size**2 if theta_lower_range is None else theta_lower_range
		self.__theta_upper_range = self.__size**2 if theta_upper_range is None else theta_upper_range

	def total_data_size (self):
		return self.__size

	def data_read (self):
		return self.__amount_generated

	def read(self):

		y = numpy.array([0.0 for i in range(0, self.__size)])
		x = numpy.array([[0.0 for j in range(0, self.__number_of_function_variables)] for i in range(0, len(y))])

		random.seed();

		for i in range(0, self.__size):
			for j in range (0, len(x[i])):
				x[i][j] = random.uniform(self.__theta_lower_range, self.__theta_upper_range)
			y[i] = self.__function(*x[i])


		return y, x

