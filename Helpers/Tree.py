# Python program to for tree traversals
import numpy as np

# A class that represents an individual node in a
# Tree
class Node:
    def __init__(self, board, isRoot=False):
        # generic tree params
        self.children = []
        self.parent = None
        self.root = None
        self.leafNode = True
        self.moves = []
        self.level = 0

        # algorithm specific
        self.board = board
        self.pruned = False
        self.stateScore = 0
        self.opponentMove = False
        self.winState = False

        #root node only
        if isRoot:
            self.descendants = []
            self.root = self



    def addChild(self, child, move):
        self.leafNode = False
        child.parent = self
        child.root = self.root
        child.moves = self.moves.copy()
        child.moves.append(move)
        child.level = len(child.moves)
        self.root.addDescendant(child)
        self.children.append(child)

    def getChildren(self):
        return self.children

    def getLevel(self):
        return self.level




    # These should really only be used for the root node
    def addDescendant(self, descendant):
        self.descendants.append(descendant)

    def getDescendants(self):
        return self.descendants

    def getLeafNodes(self):
        if len(self.descendants) == 0:
            return [self]

        return [des for des in self.descendants if des.leafNode == True]

    def getLeafNodesUnPrunedAndOpponent(self, level):
        if len(self.descendants) == 0:
            return [self]

        return [des for des in self.descendants if des.leafNode == True and des.pruned == False and len(des.moves) == level and des.opponentMove == True and des.winState == False]

    def getDepthStatistics(self):
        depths = [x.moves[-1] for x in self.descendants]
        mean = np.nanmean(depths)
        median = np.nanmedian(depths)
        sd = np.nanstd(depths)
        max = np.nanmax(depths)
        min = np.nanmin(depths)
        return (mean, median, sd, max, min)
