import numpy as np
import time
import random

#Utils

def make_move(board,move,player_number):
    """
    This function will execute the move (integer column number) on the given board, 
    where the acting player is given by player_number
    """
    newBoard = board.copy()
    row = 0
    while row < 6 and newBoard[row,move] == 0:
        row += 1
    newBoard[row-1,move] = player_number
    return newBoard

def get_valid_moves(board):
    """
    This function will return a list with all the valid moves (column numbers)
    for the input board
    """
    valid_moves = []
    for c in range(7):
        if 0 in board[:,c]:
            valid_moves.append(c)
    return valid_moves

def is_winning_state(board):
    """
    This function will tell if the player_num player is
    winning in the board that is input
    """
    player_win_str = '{0}{0}{0}{0}'.format(1)
    other_win_str = '{0}{0}{0}{0}'.format(2)
    to_str = lambda a: ''.join(a.astype(str))

    def check_horizontal(b):
        player = False
        other = False
        for row in b:
            if player_win_str in to_str(row):
                player = True
            if other_win_str in to_str(row):
                other = True
            if player and other:
                return True, True
        return player, other

    def check_verticle(b):
        return check_horizontal(b.T)

    def check_diagonal(b):
        player = False
        other = False
        for op in [None, np.fliplr]:
            op_board = op(b) if op else b
            
            root_diag = np.diagonal(op_board, offset=0).astype(np.int)
            if player_win_str in to_str(root_diag):
                player = True
            if other_win_str in to_str(root_diag):
                other = True

            for i in range(1, b.shape[1]-3):
                for offset in [i, -i]:
                    diag = np.diagonal(op_board, offset=offset)
                    diag = to_str(diag.astype(np.int))
                    if player_win_str in diag:
                        player = True
                    if other_win_str in diag:
                        other = True
                    if player and other:
                        return True, True

        return player, other

    return (check_horizontal(board)[0] or check_verticle(board)[0] or check_diagonal(board)[0],
            check_horizontal(board)[1] or check_verticle(board)[1] or check_diagonal(board)[1])

def getPlayerTurn(board):
    twos = 0
    ones = 0
    for row in board:
        for c in row:
            if c == 2:
                twos += 1
            if c == 1:
                ones += 1

    if ones > twos:
        return 2
    else:
        return 1

#The players!

class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number  #This is the id of the player this AI is in the game
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.other_player_number = 1 if player_number == 2 else 2  #This is the id of the other player

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        currentUtility = self.evaluation_function(board, 0)[0]
        print(currentUtility)

        # time limit in seconds
        TIME_LIMIT = 5
        
        Tree = Node(board, True)
        startTime = time.time()
        random.seed(startTime)

        bestMove = None;
        bestScore = -10000000000000000;
        level = 0
        while time.time() - startTime < TIME_LIMIT:
            # only explore branches where unPruned, is after opponent move, is the most recent level, and isn't an opponent win state
            leaves = Tree.getLeafNodesUnPrunedAndOpponent(level)
            while (len(leaves) == 0):
                level +=1
                leaves = Tree.getLeafNodesUnPrunedAndOpponent(level)
                if level > 1000:
                    if bestMove is None:
                        return get_valid_moves(board)[0]
                    return bestMove.moves[0]
            leafToExpand = leaves[random.randint(0, len(leaves) - 1)]
            move = self.doMyMovesAndOpponentMoves(leafToExpand, bestScore)
            if move != None:
                bestMove = move
                bestScore = self.evaluation_function(bestMove.board, bestMove.level)[0]





        stats = Tree.getDepthStatistics()
        print("Mean ", stats[0])
        print("Median ", stats[1])
        print("STD ", stats[2])
        print("Max ", stats[3])
        print("Min ", stats[4])
        validMoves = get_valid_moves(board);


        if bestMove is None:
            return validMoves[0]

        print("Best Utility ", bestMove.stateScore)
        print("Best State Depth ", bestMove.getLevel())
        currentUtility = self.evaluation_function(make_move(board, bestMove.moves[0], self.player_number), 1)[0]
        print("Current Utility ", currentUtility)
        print()
        return bestMove.moves[0]

    def doMyMovesAndOpponentMoves(self, node, bestMoveSoFarScore):
        # new bestmove
        bestScore = bestMoveSoFarScore
        newBestMove = None

        # do the branch for my move
        self.doBranch(node, self.player_number)

        # if any of the moves are a win state return that move
        for myMove in node.children:
            myMove.stateScore, myMove.winState = self.evaluation_function(myMove.board, myMove.level)
            if myMove.winState == True:
                return myMove

        # do the branches for those nodes for the opponent
        for myMove in node.children:
            opponentNewNodes = self.doBranch(myMove, self.other_player_number)
            selections = self.prune(opponentNewNodes)
            # determine if branch contains new best move
            if len(selections) > 0:
                if selections[0].stateScore > bestScore:
                    newBestMove = selections[random.randint(0, len(selections)-1)]
                    bestScore = newBestMove.stateScore



        return newBestMove


    def doBranch(self, node, player_num):
        validMoves = get_valid_moves(node.board)
        newNodes = []

        for move in validMoves:
            newBoard = make_move(node.board, move, player_num)
            newNode = Node(newBoard)
            if player_num == self.other_player_number:
                newNode.opponentMove = True
            newNodes.append(newNode)
            node.addChild(newNode, move)

        return newNodes

    def prune(self, opponentNewNodes):
        # look for nodes with the lowest score
        bestScore = 10000000
        for x in opponentNewNodes:
            x.pruned = True
            x.stateScore, x.winState = self.evaluation_function(x.board, x.level)
            if x.stateScore <= bestScore:
                bestScore = x.stateScore

        bestNodes = []
        for x in opponentNewNodes:
            if x.stateScore <= bestScore:
                bestNodes.append(x)
                x.pruned = False

        return bestNodes



    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        # do branch
        # set score of parent to average of new children
        # update scores of parent and parent of parent ...

        currentUtility = self.evaluation_function(board, 0)[0]
        print(currentUtility)

        # time limit in seconds
        TIME_LIMIT = 1

        Tree = Node(board, True)
        startTime = time.time()
        random.seed(startTime)

        bestMove = None;
        bestScore = -10000000000000000;
        level = 0
        while time.time() - startTime < TIME_LIMIT:
            # only explore branches where unPruned, is after opponent move, is the most recent level, and isn't an opponent win state
            leaves = Tree.getLeafNodesOfLevel(level)
            while (len(leaves) == 0):
                level += 1
                leaves = Tree.getLeafNodesOfLevel(level)
                if level > 1000:
                    return bestMove.moves[0]
            leafToExpand = leaves[random.randint(0, len(leaves) - 1)]
            self.doMyAndOpponentMoveExpecti(leafToExpand)

            # update score of tree
            sumLevel = level + 2
            while sumLevel > 0:
                nodesOfLevel = Tree.getNodesOfLevel(sumLevel)
                for node in nodesOfLevel:
                    if not node.leafNode:
                        node.stateScore = self.getAverageOfKids(node)
                sumLevel-=1
            Tree.stateScore = self.getAverageOfKids(Tree)

            # choose which path is the best
            for node in Tree.children:
                if node.stateScore > bestScore:
                    bestMove = node
                    bestScore = node.stateScore

        stats = Tree.getDepthStatistics()
        print("Mean ", stats[0])
        print("Median ", stats[1])
        print("STD ", stats[2])
        print("Max ", stats[3])
        print("Min ", stats[4])
        validMoves = get_valid_moves(board);

        if bestMove is None:
            return validMoves[0]

        print("Best Utility ", bestMove.stateScore)
        print("Best State Depth ", bestMove.getLevel())
        currentUtility = self.evaluation_function(make_move(board, bestMove.moves[0], self.player_number), 1)[0]
        print("Current Utility ", currentUtility)
        print()
        return bestMove.moves[0]

    def doMyAndOpponentMoveExpecti(self, node):
        # do the branch for my move
        self.doBranchExpecti(node, self.player_number)

        # do the branches for those nodes for the opponent
        for myMove in node.children:
            self.doBranchExpecti(myMove, self.other_player_number)


    def doBranchExpecti(self, node, player_num):
        validMoves = get_valid_moves(node.board)
        newNodes = []

        for move in validMoves:
            newBoard = make_move(node.board, move, player_num)
            newNode = Node(newBoard)
            newNode.stateScore = self.evaluation_function(newNode.board, newNode.level)[0]
            if player_num == self.other_player_number:
                newNode.opponentMove = True
            newNodes.append(newNode)
            node.addChild(newNode, move)

        node.stateScore = self.getAverageOfKids(node)

    def getAverageOfKids(self, node):
        total = 0
        for myMove in node.children:
            total += myMove.stateScore

        if len(node.children) == 0: return 0
        return total / len(node.children)



    def evaluation_function(self, board, depth):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """

        # count number of tiles in sequence
        # discount any that can't lead to four in a row

        def count_horizontal(b, player):
            total = 0;
            for row in b:
                total += count_InRow(row, player)

            return total

        def count_verticle(b, player):
            return count_horizontal(b.T, player)

        def count_diagonal(b, player):
            total = 0
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b

                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                total += count_InRow(root_diag, player)

                for i in range(1, b.shape[1] - 3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        total += count_InRow(diag, player)

            return total

        def count_InRow(row, player):
            op_num = 1 if player == 2 else 2
            total = 0
            s = 0;
            e = 4;
            while (e <= len(row)):
                # segment into sections of 4
                nextFour = row[s:e]
                # sum up how many in a row not blocked
                if op_num not in nextFour:
                    numInRow = np.sum(np.array(nextFour) == player)
                    #if four in a row give special value
                    if numInRow == 4:
                        numInRow *= numInRow * numInRow
                    total += numInRow

                s += 1
                e += 1

            return total

            # op_num = 1 if player == 2 else 2
            # # get sequences
            # sequences = []
            # start = None
            # for i in range(len(row)):
            #     if row[i] == player and start is None: start = i
            #     if row[i] != player and start is not None:
            #         sequences.append((start, i - 1, i - start))
            #         start = None
            #     if i + 1 == len(row) and start is not None:
            #         sequences.append((start, i, len(row) - start))
            #         start = None
            #
            # # check validity
            # validSeq = []
            # for seq in sequences:
            #
            #     i = seq[0]
            #     s = i
            #     while i > 0:
            #         i -= 1
            #         s = i
            #         if row[i] == op_num:
            #             s = i + 1
            #             i = -1
            #
            #     i = seq[0]
            #     e = i
            #     while i < len(row):
            #         if row[i] == op_num:
            #             e = i - 1
            #             i = len(row)
            #         else:
            #             i += 1
            #             e = i
            #
            #
            #     if e - s >= 4:
            #         validSeq.append(seq[2] * seq[2])
            #
            # return sum(validSeq)

        # count how many valid sequences the player has
        count = 0
        count += count_horizontal(board, self.player_number)
        count += count_verticle(board, self.player_number)
        count += count_diagonal(board, self.player_number)

        count -= count_horizontal(board, self.other_player_number)
        count -= count_verticle(board, self.other_player_number)
        count -= count_diagonal(board, self.other_player_number)



        # count how many winning possibilities the player has for next move and the move after
        validMoves = get_valid_moves(board)
        nextMoveWinningStates = 0
        firstPlayer = getPlayerTurn(board)
        winFirst = 0
        for mOp in validMoves:
            bOp = make_move(board, mOp, firstPlayer)
            if is_winning_state(bOp)[1 if firstPlayer == 2 else 0]:
                winFirst +=1

        otherPlayer = 2 if firstPlayer == 1 else 1
        winSecond = 0
        for mOp in validMoves:
            bOp = make_move(board, mOp, otherPlayer)
            if is_winning_state(bOp)[1 if otherPlayer == 2 else 0]:
                winSecond +=1

        if firstPlayer == self.player_number:
            if winFirst > 0:
                nextMoveWinningStates += winFirst
            elif winSecond > 0:
                nextMoveWinningStates -= (winSecond - 1)
        else:
            if winFirst > 0:
                nextMoveWinningStates -= winFirst
            elif winSecond > 0:
                nextMoveWinningStates += (winSecond - 1)



        count += pow(nextMoveWinningStates, 7) * 100

        winState = is_winning_state(board)
        if winState[0]:
            count += 10000000
        if winState[1]:
            count -= 10000000


        return count / (depth + 1), winState[0] or winState[1]


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move



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
        """
        RETURNS:
        The immediate children of the node but not the children's children etc..
        """
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

    def getLeafNodesOfLevel(self, level):
        if len(self.descendants) == 0:
            return [self]

        return [des for des in self.descendants if des.leafNode == True and des.level == level and des.opponentMove == True and des.winState == False]

    def getNodesOfLevel(self, level):
        if len(self.descendants) == 0:
            return [self]

        return [des for des in self.descendants if des.level == level]

    def getDepthStatistics(self):
        depths = [x.level for x in self.descendants]
        mean = np.nanmean(depths)
        median = np.nanmedian(depths)
        sd = np.nanstd(depths)
        max = np.nanmax(depths)
        min = np.nanmin(depths)
        return (mean, median, sd, max, min)
