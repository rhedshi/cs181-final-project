
import visualize


pattern = "python pacman.py -T SrcTeam -p %s -m 1000 -q"
values = [[
	"GoodBadCapsuleDistanceAgent",
	"SeededGoodBadCapsuleDistanceAgent",
	"LocalNeighborhoodAgent",
	"SeededLocalNeighborhoodAgent",
	"GhostPositionAgent",
	"SafeAgent",
	"BadGhostAgent",
	"CapsuleAgent",
	"GhostAgent",
	"NearestBadGhostAgent"
]]

visualize.savePlots("plots/agents-trajectories.pdf",[visualize.plotScores(*visualize.compareScores(pattern, *values), fmt="%s")])
visualize.savePlots("plots/agents-finals.pdf",[visualize.plotFinalScores(*visualize.compareFinalScores(pattern, *values, n=10), fmt="%s")])