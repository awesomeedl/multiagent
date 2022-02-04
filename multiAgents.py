# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from sre_constants import SUCCESS
from util import manhattanDistance
from game import Directions, GameStateData
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        ret = successorGameState.getScore()

        if newFood.count() > 0:
            distanceToFood = min(map(lambda food: abs(food[0] - newPos[0]) + abs(food[1] - newPos[1]), newFood.asList() ))
            ret += 1.0 / float(distanceToFood)

        # distanceToGhost = min(map(lambda ghost: abs(ghost.getPosition()[0] - newPos[0]) + abs(ghost.getPosition()[1] - newPos[1]), newGhostStates ))
        # if distanceToGhost > 0:
        #     ret += 1.0 - ( 1.0 / distanceToGhost )
        
        return ret

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        return max(gameState.getLegalActions(0), key= lambda action: self.miniMaxHelper(gameState.generateSuccessor(0, action), depth=1, agent=1))
        

    def miniMaxHelper(self, gameState, depth, agent):
        # Base case
        if depth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState) # Utility for leaf
        
        successorStates = [gameState.generateSuccessor(agent, action) for action in gameState.getLegalActions(agent)] # All possible successors
        
        v = -2**31 - 1 if agent == 0 else 2**31 - 1 # Initial utility as infinity
        for successorState in successorStates:
            if agent == 0:  # Pacman
                v = max(v, self.miniMaxHelper(successorState, depth, 1))
            elif agent == gameState.getNumAgents() - 1: # Last Ghost, next round is pacman
                v = min(v, self.miniMaxHelper(successorState, depth + 1, 0))
            else:   # Ghost
                v = min(v, self.miniMaxHelper(successorState, depth, agent + 1))
                
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    alpha = beta = None

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        # legalActions = gameState.getLegalActions(0)
        # successorStates = [gameState.generateSuccessor(0, action) for action in legalActions]
        return max(gameState.getLegalActions(0), key= lambda action: self.alphaBetaHelper(gameState.generateSuccessor(0, action), depth=1, agent=1))

    def alphaBetaHelper(self, gameState, depth, agent):
        # Base case
        if depth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState) # Utility for leaf
        
        successorStates = [gameState.generateSuccessor(agent, action) for action in gameState.getLegalActions(agent)] # All possible successors
        
        v = -2**31 - 1 if agent == 0 else 2**31 - 1 # Initial utility as infinity
        for successorState in successorStates:
            if agent == 0:  # Pacman
                v = max(v, self.alphaBetaHelper(successorState, depth, 1))
                if self.beta != None and v > self.beta:
                    return v
                self.alpha = max(self.alpha, v) if self.alpha != None else v
            elif agent == gameState.getNumAgents() - 1: # Last Ghost, next round is pacman
                v = min(v, self.alphaBetaHelper(successorState, depth + 1, 0))
                if self.alpha != None and v < self.alpha:
                    return v
                self.beta = min(self.beta, v) if self.beta != None else v
            else:   # Ghost
                v = min(v, self.alphaBetaHelper(successorState, depth, agent + 1))
                if self.alpha != None and v < self.alpha:
                    return v
                self.beta = min(self.beta, v) if self.beta != None else v     

        print("Alpha: " + str(self.alpha) + " Beta: " + str(self.beta))
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
