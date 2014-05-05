import visualize
import sys

if sys.argv < 2:
	raise "Need agent name"

agentName = sys.argv[1]
repeats = 50
pattern = "python pacman.py -T SrcTeam -p %s -m 1000 -q"
values = [[	agentName ] * repeats]
visualize.savePlots("plots/train-%s.pdf" % agentName,[visualize.plotFinalScores(*visualize.compareFinalScores(pattern, *values, n=5), fmt="%s")])
