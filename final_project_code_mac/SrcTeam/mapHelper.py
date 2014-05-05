from game import Directions
import random

def between(pt1, pt2, pt3):
    "returns true if pt3 is between pt2 and pt1"
    return sameSign(pt3[0] - pt2[0], pt2[0] - pt1[0]) and sameSign(pt3[1] - pt2[1], pt2[1] - pt1[1])

def sameSign(i1, i2):
    "returns true if i1 and i2 are same sign. 0 is both pos and neg"
    return (i1 <= 0 and i2 <= 0) or (i1 >= 0 and i2 >= 0)

def subTarget(pacman, target, subTarget):
    """args: positions of pacman, target (eg. capsule), and a subTarget (eg. ghost)
    if subTarget is on the way to target, then returns subTarget"""
    if between(pacman, target, subTarget):
        return subTarget
    else:
        return target

def ghostDirs(pacman, ghost):
    """args: positions of pacman and the ghost
    returns a list of directions that may lead to the ghost"""
    xDiff = ghost[0] - pacman[0]
    yDiff = ghost[1] - pacman[1]
    dirs = []

    if (xDiff == 1 and abs(yDiff) <=1) or (xDiff == 2 and yDiff == 0):
        dirs.append(Directions.EAST)
    elif (xDiff == -1 and abs(yDiff) <=1) or (xDiff == -2 and yDiff == 0):
        dirs.append(Directions.WEST)
    elif yDiff > 0 and yDiff <=2 and xDiff == 0:
        dirs.append(Directions.NORTH)
    elif yDiff < 0 and yDiff >= -2 and xDiff == 0:
        dirs.append(Directions.SOUTH)

    return dirs

def getDirs(pt1, pt2):
    "returns a list of possible directions to get from pt1 to pt2"
    dirs = []

    # if pt2[1] > pt1[1]:
    #     dirs.append(Directions.NORTH)
    # elif pt2[1] < pt1[1]:
    #     dirs.append(Directions.SOUTH)

    # if pt2[0] > pt1[0]:
    #     dirs.append(Directions.EAST)
    # elif pt2[0] < pt1[0]:
    #     dirs.append(Directions.WEST)

    x = random.randint(0,1)
    if pt2[x] > pt1[x]:
        if x == 1:
            dirs.append(Directions.NORTH)
        else:
            dirs.append(Directions.EAST)
    elif pt2[x] < pt1[x]:
        if x == 1:
            dirs.append(Directions.SOUTH)
        else:
            dirs.append(Directions.WEST)

    if pt2[1-x] > pt1[1-x]:
        if x == 0:
            dirs.append(Directions.NORTH)
        else:
            dirs.append(Directions.EAST)
    elif pt2[1-x] < pt1[1-x]:
        if x == 0:
            dirs.append(Directions.SOUTH)
        else:
            dirs.append(Directions.WEST)
    return dirs