import numpy as np

def unpickle(file):
	"""
	Loads and returns a pickled data structure in the given `file` name
	"""
	import cPickle
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict

def pickle(array, file):
	"""
	Dumps an array to a file
	"""
	import cPickle
	fo = open(file,'wb')
	cPickle.dump(array,fo)
	fo.close()

def csv_to_np_array(file):
	return np.genfromtxt(file, delimiter=" ", names=None)