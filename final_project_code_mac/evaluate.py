import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import visualize


pattern = "python -W ignore pacman.py -T SrcTeam -p %s -m 1000 -q"
values = [[
	"GoodBadCapsuleDistanceAgent"
	, "SeededGoodBadCapsuleDistanceAgent"
	, "LocalNeighborhoodAgent"
	, "SeededLocalNeighborhoodAgent"
	, "GhostPositionAgent"
	, "SafeAgent"
	, "BadGhostAgent"
	, "CapsuleAgent"
	, "GhostAgent"
	, "AnyGhostAgent"
]]

n = 10
visualize.savePlots("plots/agents-trajectories.pdf",[visualize.plotScores(*visualize.compareScores(pattern, *values), fmt="%s")])
visualize.savePlots("plots/agents-finals.pdf",[visualize.plotFinalScores(*visualize.compareFinalScores(pattern, *values, n=n, seeds=range(1,n+1)), fmt="%s")])