from os import cpu_count

def execute_threads(thread_queue, join_function):
	processors_available = cpu_count()
	processors_needed = processors_available if len(thread_queue) > processors_available else len(thread_queue)

	threads_running = []
	for i in range(processors_needed):
		thread = thread_queue.pop(0)
		thread.start()
		threads_running.append(thread)

	while len(threads_running) > 0:
		current_thread = threads_running.pop(0)

		current_thread.join()
		join_function(current_thread)

		if len(thread_queue) > 0:
			thread = thread_queue.pop(0)
			thread.start()
			threads_running.append(thread)
