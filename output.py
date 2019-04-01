from exceptions import AbstractMethodCallError

class Observable():
	def __init__(self):
		self.__observers = []

	def attach (self, observer):
		self.__observers.append(observer)

	def notify_observers (self):
		for observer in self.__observers: observer.update(self)

class Observer ():
	def update (self, data):
		raise AbstractMethodCallError("update")

class DatasetReaderObserver (Observer):
	def __init__(self, output):
		self.__reading_has_begun = False
		if (not output.writable()):
			raise RuntimeError("Need writable output to create observer")
		self.__output = output

	def update(self, reader):
		progress = reader.data_already_read()/reader.total_data_size()*100

		if (not self.__reading_has_begun):
			self.__reading_has_begun = True
			self.__output.write("\nReading dataset...\n")
		else:
			self.__output.write("\033[1A\r")

		self.__output.write("{0:.1f}%\n".format(progress))

		if progress >= 100:
			self.__output.write("Dataset read successfuly\n")
			self.__reading_has_begun = False

class TrainerObserver (Observer):
	def __init__(self, output):
		self.__training_has_begun = False
		if (not output.writable()):
			raise RuntimeError("Need writable output to create observer")
		self.__output = output

	def update(self, trainer):

		if trainer.is_trained():
			self.__output.write("Training complete\n")
			self.__training_has_begun = False
		else:
			if (not self.__training_has_begun):
				self.__training_has_begun = True
				self.__output.write("\nTraining...\n")
			else:
				self.__output.write("\033[5A\r")

			self.__output.write("Iteration:\t\t" + str(trainer.get_current_iteration()) + "              \n")
			self.__output.write("Relative error:\t\t" + str(trainer.get_relative_error()) + "              \n")
			self.__output.write("Absolute error:\t\t" + str(trainer.get_absolute_error()) + "              \n")
			self.__output.write("Convergence rate:\t{0:.2f}%".format(100.0*trainer.get_convergence_rate()) + "              \n")
			self.__output.write("Learning bias:\t\t" + str(trainer.get_learning_bias()) + "              \n")

