import time
from math import floor

class Timer():
	def __init__(self):
		self.__number_of_ticks = 0
		self.__total_time = 0
		self.__last_tick_interval = 0

	def begin(self):
		self.__begin_time = time.clock()

	def tick(self):
		current_time = time.clock()

		self.__number_of_ticks += 1

		self.__last_tick_interval = current_time - self.__begin_time
		self.__begin_time = current_time

		self.__total_time += self.__last_tick_interval


	def __formatted_time(self, time):
		days = floor(time/86400)
		time -= days*86400
		hours = floor(time/3600)
		time -= hours*3600
		minutes = floor(time/60)
		time -= minutes*60
		seconds = floor(time)
		time -= seconds
		milliseconds = floor(time*1000)

		formatted_time = ""
		if days > 0:
			formatted_time += str(days) + "d"
		if hours > 0:
			if len(formatted_time) > 0:
				formatted_time += " "
			formatted_time += str(hours) + "h"
		if minutes > 0:
			if len(formatted_time) > 0:
				formatted_time += " "
			formatted_time += str(minutes) + "m"
		if seconds > 0:
			if len(formatted_time) > 0:
				formatted_time += " "
			formatted_time += str(seconds) + "s"
		if milliseconds > 0:
			if len(formatted_time) > 0:
				formatted_time += " "
			formatted_time += str(milliseconds) + "ms"

		if len(formatted_time) == 0:
			formatted_time = "0ms"

		return formatted_time

	def get_average_tick_interval(self):
		if self.__number_of_ticks > 0:
			time = self.__total_time/self.__number_of_ticks
		else:
			time = 0
		return self.__formatted_time(time)

	def get_total_elapsed_time(self):
		return self.__formatted_time(self.__total_time)

	def get_last_tick_interval(self):
		return self.__formatted_time(self.__last_tick_interval)
