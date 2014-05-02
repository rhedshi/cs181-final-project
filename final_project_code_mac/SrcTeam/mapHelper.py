from game import Directions

def between(pt1, pt2, pt3):
    "returns true if pt3 is between pt2 and pt1"
    return sameSign(p3[0] - pt2[0], pt2[0] - pt1[0]) and sameSign(p3[1] - pt2[1], pt2[1] - pt1[1])

def sameSign(i1, i2):
    "returns true if i1 and i2 are same sign. 0 is both pos and neg"
    return (i1 <= 0 and i2 <= 0) or (i1 >= 0 and i2 >= 0)

def getDirs(pt1, pt2):
    "returns a list of possible directions to get from pt1 to pt2"
    dirs = []
    if pt2[1] > pt1[1]:
        dirs.append(Directions.NORTH)
    elif pt2[1] < pt1[1]:
        dirs.append(Directions.SOUTH)

    if pt2[0] > pt1[0]:
        dirs.append(Directions.EAST)
    elif pt2[0] < pt1[0]:
        dirs.append(Directions.WEST)

    return dirs