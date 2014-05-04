import itertools as it
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def getScores(command):
	out = subprocess.check_output(command, shell=True)
	lines = out.split("\n")
	scorePattern = re.compile(r"score: (-?\d+\.?\d+)")
	scores = [ float(scorePattern.match(line).group(1)) for line in lines if line.strip() != "" and scorePattern.match(line) ]
	return scores

def compareScores(pattern, *values):
	scores = list()
	combinations = list(it.product(*values))
	for val in combinations:
		cmd = pattern % val
		scores.append(getScores(cmd))

	return combinations, scores

def compareFinalScores(pattern, *values, **kwargs):
	n = kwargs['n'] if 'n' in kwargs else 5
	scores = list()
	combinations = list(it.product(*values))
	for val in combinations:
		cmd = pattern % val
		scores.append(getFinalScores(cmd, n))

	return combinations, scores

def getFinalScores(command, n=5):
	return [ getScores(command)[-1] for i in xrange(n)]

def plotFinalScores(values, scores, fmt=None):
	plt.figure()
	ascores = np.array(scores).T
	ind = np.arange(len(scores))
	width = 0.8
	plt.bar( ind, np.mean(ascores, axis=0), width, yerr=np.std(ascores) )
	if fmt != None:
		labels = tuple( fmt % val for val in values)
	else:
		labels = tuple(map(str,values))
	plt.xticks(ind+width/2., labels, rotation=70)
	plt.draw()
	plt.tight_layout()
	return plt.gcf()

def plotScores(values, scores, fmt=None):
	plt.figure()
	ax = plt.subplot(111)
	plt.plot(np.array(scores).T)
	if fmt != None:
		labels = tuple( fmt % val for val in values)
	else:
		labels = tuple(map(str,values))

	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
	plt.legend(labels, loc='center left',bbox_to_anchor=(1, 0.5),prop={'size':6})
	plt.draw()
	# plt.tight_layout()

	return plt.gcf()

def savePlots(filename, figs):
	pp = PdfPages(filename)
	for fig in figs:
		pp.savefig(fig)
	pp.close()

