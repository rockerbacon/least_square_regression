from os import cpu_count
from threading import Thread, Lock, Condition, currentThread
from exceptions import AbstractMethodCallError
from time import sleep

class ManagedFunction (Thread):
	def __init__(self, function, args, manager):
		Thread.__init__(self)
		self.__function = function
		self.__args = args
		self.__manager = manager

	def run(self):
		return_value = self.__function(*self.__args)
		self.__manager.notify_return(return_value)

	def get_return (self):
		return self.__return

class ThreadManager ():
	def __init__(self, cpu_cores=None):
		self.__cpu_cores = cpu_count() if cpu_cores is None else cpu_cores

		self.__functions = []
		self.__function_queue = []
		self.__function_queue_lock = Lock()

		self.__all_threads_finished_condition = Condition()

		self.__returned_values = []


	def attach(self, function, args):
		self.__functions.append({"function": function, "args": args})

	def notify_return(self, return_value):

		with self.__function_queue_lock:

			self.__returned_values.append(return_value)

			if len(self.__function_queue) > 0:
				thread = self.__function_queue.pop(0)
				thread.start()
			elif len(self.__returned_values) == len(self.__functions):
				with self.__all_threads_finished_condition:
					self.__all_threads_finished_condition.notify()

	def execute_all (self):
		with self.__all_threads_finished_condition:
			self.__function_queue = [ManagedFunction(f["function"], f["args"], self) for f in self.__functions]
			for i in range(self.__cpu_cores):
				thread = self.__function_queue.pop(0)
				thread.start()

			self.__all_threads_finished_condition.wait()

		return self.__returned_values
