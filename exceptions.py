class AbstractMethodCallError(Exception):
	def __init__(self, method_name):
		super().__init__("Attempted to call abstract method "+method_name+"() from a class which does not provide a concrete implementation")
