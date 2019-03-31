from exceptions import AbstractMethodCallError

class Observer ():
	def update (self, data):
		raise AbstractMethodCallError("update")

class DatasetReadingObserver (Observer):
	def __init__(self, output):
		self.__reading_has_begun = False
		if (not output.writable()):
			raise RuntimeError("Need writable output to create observer")
		self.__output = output

	def update(self, reader):
		progress = reader.data_already_read()/reader.total_data_size()*100

		if (not self.__reading_has_begun):
			self.__reading_has_begun = True
			self.__output.write("\nReading dataset...\n0%\n")

		self.__output.write("\033[1A\r{0:.1f}%\n".format(progress))

		if progress >= 100:
			self.__output.write("Dataset read successfuly\n")
			self.__reading_has_begun = False
