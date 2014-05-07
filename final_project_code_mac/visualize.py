import itertools as it
import re
import random
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
	seeds = kwargs['seeds'] if 'seeds' in kwargs else None
	scores = list()
	combinations = list(it.product(*values))
	for val in combinations:
		cmd = pattern % val
		print '--------------------------------------------------------------------------------'
		print cmd
		scores.append(getFinalScores(cmd, n=n, seeds=seeds))

	return combinations, scores

def getFinalScores(command, n=5, seeds=None):
	seeds = seeds if seeds != None else [random.randint(1,10) for i in xrange(n)]
	assert len(seeds) == n
	return [ getScores(command + " -s " + str(seeds[i]))[-1] for i in xrange(n)]

def plotFinalScores(values, scores, fmt=None):
	plt.figure()
	ascores = np.array(scores).T
	ind = np.arange(len(scores))
	width = 0.8
	ax = plt.subplot(111)
	ax.set_color_cycle(["#ff4040", "#ff6600", "#ffb380", "#ffaa00", "#ffee00", "#88ff00", "#d9ffbf", "#80ffb2", "#80fff6", "#40d9ff", "#00aaff", "#bfeaff", "#4073ff", "#bfbfff", "#6600ff", "#d580ff", "#ff00ee", "#ff40a6", "#ffbfe1", "#ff8091"])
	plt.bar( ind, np.mean(ascores, axis=0), width, yerr=np.std(ascores, axis=0) )
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
	ax.set_color_cycle(["#ff4040", "#ff6600", "#ffb380", "#ffaa00", "#ffee00", "#88ff00", "#d9ffbf", "#80ffb2", "#80fff6", "#40d9ff", "#00aaff", "#bfeaff", "#4073ff", "#bfbfff", "#6600ff", "#d580ff", "#ff00ee", "#ff40a6", "#ffbfe1", "#ff8091"])
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


