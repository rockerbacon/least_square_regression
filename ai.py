from sklearn.base import BaseEstimator
import math
import numpy
import sys
import warnings
import threading
import random
from output import Observable
import thread_manager

#function theta[0] + theta[1]*x[0] + theta[2]*x[1] + ... + theta[n+1]*x[n]
def linear_function (x, theta):
	result = theta[0] + sum ( [ theta[n+1]*x[n] for n in range(len(x)) ] )
	return result

#function theta[0] + theta[1]*x[0]^1 + theta[2]*x[1]^2 + ... + theta[n+1]*x[n]^(n+1)
def polinomial_function (x, theta):
	result = theta[0] + sum ([ theta[n+1]*x[n]**(n+1) for n in range(len(x)) ])
	return result

def minibatch_error_evaluation_function (y, x, theta, prediction_function):
	error = 0
	setsize = math.floor(len(y)*0.3)
	for i in range (setsize):
		error += abs(y[i] - prediction_function(x[i], theta))
	error /= setsize

	return error

def batch_error_evaluation_function (y, x, theta, prediction_function):
	error = 0
	for i in range(len(y)):
		error += abs(y[i] - prediction_function(x[i], theta))
	error /= len(y)

	return error

def relative_error_stop_function (trainer):
	return trainer.get_relative_error() < 0.0001

def absolute_error_stop_function (trainer):
	return trainer.get_absolute_error() < 0.0001

class DerivedCostEvaluator (threading.Thread):
	def __init__ (self, theta_index, y, x, theta, prediction_function):
		threading.Thread.__init__(self)
		self.__theta_index = theta_index
		self.__y = y
		self.__x = x
		self.__theta = theta
		self.__predict = prediction_function

	#derivative of the euclidean distance
	def run (self):
		if self.__theta_index > 0:
			self.__derived_cost = sum( [ (self.__predict(self.__x[i], self.__theta) - self.__y[i])*self.__x[i][self.__theta_index-1] for i in range(len(self.__x)) ] )
		else:
			self.__derived_cost = sum( [ (self.__predict(self.__x[i], self.__theta) - self.__y[i]) for i in range(len(self.__x)) ] )

	def get_derived_cost(self):
		return self.__derived_cost

	def get_theta_index(self):
		return self.__theta_index

class LMSTrainer(BaseEstimator, Observable):
	def __init__(self):
		super().__init__()

		self.__current_iteration = 0
		self.__absolute_error = float("inf")
		self.__relative_error = float("inf")
		self.__convergence_rate = 0

		self.__trained = False

	def set_prediction_function (self, function):
		self.__predict = function

	def set_error_evaluation_function (self, function):
		self.__evaluate_error = function

	def set_stop_function (self, function):
		self.__stop = function

	def set_trainingset (self, trainingset_y, trainingset_x):
		self.__y = trainingset_y
		self.__x = trainingset_x

	def set_initial_theta_values (self, theta):
		self.__theta = theta
		self.__theta_derived_cost = numpy.array([float("inf") for i in range(len(self.__theta))])

	def set_learning_bias (self, rate):
		self.__learning_bias = rate

	def set_adaptive_learning (self, use_adaptive_learning):
		self.__adaptive_learning = use_adaptive_learning

	def set_epochs (self, number_of_epochs):
		self.__epochs = number_of_epochs

	def set_analitic (self, analitic):
		self.__analitic = analitic

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

	def is_trained (self):
		return self.__trained

	#TODO verify function
	def __adjust_learning (self):
		if self.__convergence_rate > 0.0:
			#attempt to find the highest possible value for the learning rate
			self.__learning_bias = self.__learning_bias*increasingRate
			increasingRate = (2.0-learningAdaptionRate)*(increasingRate**2.0 - 2.0*increasingRate + 2.0) - (1.0-learningAdaptionRate)	#reduce climbing speed the more you climb

		else:
			#after climbing begin to reduce learning value to better approximate the result
			learningRate = learningRate*decreasingRate

			#reduce learning rate according to convergence rate. The bigger the improvements the more likely the next improvements will come from small steps
			decreasingRate = learningAdaptionRate*(-decreasingRate**2.0 + 2.0*decreasingRate) + (1.0 - learningAdaptionRate) #reduce decline speed the more you decline


	def __join_thread (self, thread):
		self.__theta_derived_cost[thread.get_theta_index()] = thread.get_derived_cost()

	def __evaluate_derived_cost(self):

		thread_queue = []
		for theta_index in range(len(self.__theta)):
			thread = DerivedCostEvaluator(theta_index, self.__y, self.__x, self.__theta, self.__predict)
			thread_queue.append(thread)

		thread_manager.execute_threads(thread_queue, self.__join_thread)

	def fit(self):

		self.notify_observers()

		if self.__analitic:
			# TODO: FAZER POR MATRIZES
			pass
		else:

			while not self.__stop(self):
				self.__current_iteration += 1

				for ep in range(self.__epochs):

					self.__evaluate_derived_cost()

					previous_error = self.__absolute_error
					self.__absolute_error = self.__evaluate_error(self.__y, self.__x, self.__theta, self.__predict)
					self.__relative_error = previous_error - self.__absolute_error

					if self.__adaptive_learning:
						self.__adjust_learning()

					#update theta values
					for i in range(len(self.__theta_derived_cost)):
						self.__theta[i] -= self.__learning_bias*self.__theta_derived_cost[i]

				self.notify_observers()

		self.__trained = True
		self.notify_observers()

		return self

	def predict(self, x, y=None):

		if not self.is_trained():
			raise RuntimeError("Method fit() must be called before method predict")

		prediction = self.__predict(x, self.__theta)

		return prediction

class LMSTrainerBuilder ():
	def __init__(self):
		self.__trainer = LMSTrainer()

	def build (self):
		return self.__trainer

	def with_defaults (self, trainingset_y, trainingset_x):
		random.seed()
		self.__trainer.set_prediction_function(linear_function)
		self.__trainer.set_stop_function(relative_error_stop_function)
		self.__trainer.set_trainingset(trainingset_y, trainingset_x)
		self.__trainer.set_initial_theta_values(numpy.array([random.uniform(-10, 10) for i in range(len(trainingset_x[0])+1)]))
		self.__trainer.set_learning_bias(0.000002)
		self.__trainer.set_adaptive_learning(False)
		self.__trainer.set_epochs(1)
		self.__trainer.set_analitic(False)
		self.__trainer.set_error_evaluation_function(minibatch_error_evaluation_function)
		return self

	def with_prediction_function(self, function):
		self.__trainer.set_prediction_function(function)
		return self

	def with_error_function(self, function):
		self.__trainer.set_derived_cost_function(function)
		return self

	def with_stop_function(self, function):
		self.__trainer.set_stop_function(function)
		return self

	def with_trainingset(self, trainingset_y, trainingset_x):
		self.__trainer.set_trainingset(trainingset_y, trainingset_x)
		return self

	def with_initial_theta_value(self, theta):
		self.__trainer.set_initial_theta_values(theta)
		return self

	def with_learning_bias(self, bias):
		self.__trainer.set_learning_bias(bias)
		return self

	def with_adaptive_learning(self, adaptive):
		self.__trainer.set_adaptive_learning(adaptive)
		return self

	def with_epochs(self, epochs):
		self.__trainer.set_epochs(epochs)
		return self

	def with_analitic (self, analitic):
		self.__trainer.set_analitic(analitic)
		return self

	def with_error_evaluation_function (self, function):
		self.__trainer.set_error_evaluation_function(function)
		return self

