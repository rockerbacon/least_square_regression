from sklearn.base import BaseEstimator
import math
import numpy
import random
from output import Observable
from thread_manager import ThreadManager
from clock import Timer

class CostFunction():
	def prediction(self, x, theta):
		raise AbstractMethodCallError("function")

	def cost_gradient(self, dataset_y, dataset_x, theta, theta_index):
		raise AbstractMethodCallError("derivative")

class LinearCostFunction(CostFunction):
	def __init__(self):
		super().__init__()

	def prediction(self, x, theta):
		result = theta[0] + sum ( [ theta[n+1]*x[n] for n in range(len(x)) ] )
		return result

	def cost_gradient(self, dataset_y, dataset_x, theta, theta_index):
		if theta_index > 0:
			cost = sum ([ (self.prediction(dataset_x[n], theta) - dataset_y[n])*dataset_x[n][theta_index-1] for n in range(len(dataset_x)) ]) / len(dataset_x)
		else:
			cost = sum ([ self.prediction(dataset_x[n], theta) - dataset_y[n] for n in range(len(dataset_x)) ]) / len(dataset_x)

		return {"theta_index": theta_index, "derived_cost": cost}

class PolinomialCostFunction(CostFunction):
	def __init__(self):
		super().__init__()

	def prediction(self, x, theta):
		result = theta[0] + sum ( [ theta[n+1]*x[n]**(n+1) for n in range(len(x)) ] )
		return result

	def cost_gradient(self, dataset_y, dataset_x, theta, theta_index):
		if theta_index > 0:
			cost = sum ([ (self.prediction(dataset_x[n], theta) - dataset_y[n])*dataset_x[n][theta_index-1] for n in range(len(dataset_x)) ]) / len(dataset_x)
		else:
			cost = sum ([ (self.prediction(dataset_x[n], theta) - dataset_y[n])*(theta_index-1)*dataset_x[n][theta_index-1]**(theta_index-2) for n in range(len(dataset_x)) ]) / len(dataset_x)

		return {"theta_index": theta_index, "derived_cost": cost}

def evaluate_error(y, x, theta, cost_function):
	return abs(cost_function.prediction(x, theta) - y)

def minibatch_error_evaluation_function (y, x, theta, cost_function, percentage_to_evaluate=0.3):

	setsize = math.floor(len(y)*percentage_to_evaluate)

	thread_manager = ThreadManager(1)
	for i in range (setsize):
		thread_manager.attach(evaluate_error, (y[i], x[i], theta, cost_function))

	errors = thread_manager.execute_all()

	error = sum(errors)/setsize

	return error

def batch_error_evaluation_function (y, x, theta, cost_function):
	error = 0
	for i in range(len(y)):
		error += abs(y[i] - cost_function.prediction(x[i], theta))
	error /= len(y)

	return error

def relative_error_stop_function (trainer, threshold=0.0001):
	return trainer.get_relative_error() < threshold

def absolute_error_stop_function (trainer, threshold=0.0001):
	return trainer.get_absolute_error() < threshold

def numberof_iterations_stop_function (trainer, iterations=5):
	return trainer.get_current_iteration() >= iterations

class LMSTrainer(BaseEstimator, Observable):
	def __init__(self):
		super().__init__()

		self.__current_iteration = 0
		self.__absolute_error = float("inf")
		self.__relative_error = float("inf")
		self.__convergence_rate = 0

		self.__trained = False

		self.__timer = Timer()

	def set_error_function (self, function):
		self.__evaluate_error = function

	def set_stop_function (self, function):
		self.__stop = function

	def set_trainingset (self, trainingset_y, trainingset_x):
		self.__y = trainingset_y
		self.__x = trainingset_x

	def set_initial_theta_values (self, theta):
		self.__theta = theta

	def set_learning_bias (self, rate):
		self.__learning_bias = rate

	def set_adaptive_learning (self, use_adaptive_learning):
		self.__adaptive_learning = use_adaptive_learning

	def set_epochs (self, number_of_epochs):
		self.__epochs = number_of_epochs

	def set_cost_function (self, cost_function):
		self.__cost_function = cost_function

	def get_current_iteration (self):
		return self.__current_iteration

	def get_absolute_error (self):
		return self.__absolute_error

	def get_relative_error (self):
		return self.__relative_error

	def get_convergence_rate (self):
		return self.__convergence_rate

	def get_learning_bias (self):
		return self.__learning_bias

	def get_timer(self):
		return self.__timer

	def is_trained (self):
		return self.__trained

	def __adjust_learning (self):
		self.__learning_bias *= 1.2*math.exp(self.__convergence_rate)

	def __evaluate_gradient(self):

		thread_manager = ThreadManager(1)
		for theta_index in range(len(self.__theta)):
			thread_manager.attach(self.__cost_function.cost_gradient, (self.__y, self.__x, self.__theta, theta_index))

		return thread_manager.execute_all()

	def fit(self, analitic=False):

		self.notify_observers()
		self.__timer.begin()

		if analitic:
			# TODO: FAZER POR MATRIZES
			pass
		else:

			while not self.__stop(self):
				self.__current_iteration += 1

				for ep in range(self.__epochs):

					costs = self.__evaluate_gradient()

					previous_error = self.__absolute_error
					self.__absolute_error = self.__evaluate_error(self.__y, self.__x, self.__theta, self.__cost_function)
					self.__relative_error = abs(previous_error - self.__absolute_error)
					self.__convergence_rate = 1 - self.__absolute_error/previous_error

					if self.__adaptive_learning:
						self.__adjust_learning()

					#update theta values
					for cost in costs:
						self.__theta[cost["theta_index"]] -= self.__learning_bias*cost["derived_cost"]

					self.__timer.tick()
					self.notify_observers()

		self.__trained = True
		self.notify_observers()

		return self

	def predict(self, x, y=None):

		if not self.is_trained():
			raise RuntimeError("Method fit() must be called before method predict")

		prediction = self.__cost_function.prediction(x, self.__theta)

		return prediction

class LMSTrainerBuilder ():
	def __init__(self):
		self.__trainer = LMSTrainer()
		self.__missing_obrigatory_parameters = {'trainingset', 'learning_bias'}
		self.__default_optional_parameters =	{
												self.with_stop_function: [relative_error_stop_function],
												self.with_random_theta_values: [0, 0],
												self.with_adaptive_learning: [True],
												self.with_epochs: [1],
												self.with_error_function: [minibatch_error_evaluation_function],
												self.with_cost_function: [LinearCostFunction()]
												}

	def __get_missing_obrigatory_parameters(self):
		missing = ""
		for p in self.__missing_obrigatory_parameters:
			missing += " " + p
		return missing

	def build (self):
		missing_obrigatory_parameters = self.__get_missing_obrigatory_parameters()
		if len(missing_obrigatory_parameters) > 0:
			raise RuntimeError("Cannot build LMSTrainer because the following obrigatory parameters were not specified:"+missing_parameters)

		default_parameters = self.__default_optional_parameters.copy()
		for with_missing_optional_parameter, default_value in default_parameters.items():
			with_missing_optional_parameter(*default_value)

		return self.__trainer

	def with_stop_function(self, function):
		self.__trainer.set_stop_function(function)
		self.__default_optional_parameters.pop(self.with_stop_function, None)
		return self

	def with_trainingset(self, trainingset_y, trainingset_x):
		self.__trainer.set_trainingset(trainingset_y, trainingset_x)
		self.__number_of_variables = len(trainingset_x[0])+1
		self.__missing_obrigatory_parameters.discard('trainingset')
		return self

	def with_initial_theta_values(self, theta):
		self.__trainer.set_initial_theta_values(theta)
		self.__default_optional_parameters.pop(self.with_random_theta_values, None)
		return self

	def with_random_theta_values(self, lower_range, upper_range, number_of_variables=None):
		if number_of_variables is None:
			if 'trainingset' in self.__missing_obrigatory_parameters:
				raise RuntimeError("Cannot infer number of variables for theta. Either specify the training set first or pass the number of variables as an argument")
			else:
				number_of_variables = self.__number_of_variables

		random.seed()
		return self.with_initial_theta_values(numpy.array([random.uniform(lower_range, upper_range) for i in range(number_of_variables)]))

	def with_learning_bias(self, bias):
		self.__trainer.set_learning_bias(bias)
		self.__missing_obrigatory_parameters.discard('learning_bias')
		return self

	def with_adaptive_learning(self, adaptive):
		self.__trainer.set_adaptive_learning(adaptive)
		self.__default_optional_parameters.pop(self.with_adaptive_learning, None)
		return self

	def with_epochs(self, epochs):
		self.__trainer.set_epochs(epochs)
		self.__default_optional_parameters.pop(self.with_epochs, None)
		return self

	def with_error_function (self, function):
		self.__trainer.set_error_function(function)
		self.__default_optional_parameters.pop(self.with_error_function, None)
		return self

	def with_cost_function (self, cost_function):
		self.__trainer.set_cost_function(cost_function)
		self.__default_optional_parameters.pop(self.with_cost_function, None)
		return self

