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
# James Kauffman (u1466212)
# Tony Zhang (u1183156)


from util import manhattanDistance
from game import Directions
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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #Ideas:
        #USE A WEIGHTED LINEAR SUM OF THE FOLLOWING:
        #Wether the move results in a win or a loss (the former the better, the latter the worse)
        #Distance to nearest ghost (the further the better, unless the ghost is scared) 
        #Distance to nearest food (the closer the better)
        #Number of remaining food pellets (the fewer the better)
        #Number of remaining power pellets (the fewer the better)
        
        if successorGameState.isWin():
            return float('inf')
        if successorGameState.isLose():
            return float('-inf')
        
        
        score = successorGameState.getScore()
        
        if newFood:
            minFoodDistance = min([manhattanDistance(newPos, food) for food in newFood])
            foodScore = 1.0 / (minFoodDistance + 1)  # Closer food increases score
        else:
            foodScore = 0
        
        ghostScore = 0.0
        for g,t in zip(newGhostStates,newScaredTimes):
            ghostDistance = manhattanDistance(newPos, g.getPosition())
            if t > 0:  # Ghost is scared
                if ghostDistance > 0:
                    ghostScore += 200.0 / ghostDistance  # Closer scared ghost increases score
            else:  # Ghost is not scared
                if ghostDistance > 0:
                    ghostScore -= 10.0 / ghostDistance  # Closer active ghost decreases score
                else:
                    return float('-inf')  # Collision with active ghost is the worst
        
        scoreWeight = 5
        foodWeight = 10
        ghostWeight = 1
    
        return (scoreWeight*score + foodWeight*foodScore + ghostWeight*ghostScore)
        
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
        #Ideas:
        #Implement the minimax algorithm with getNumAgents()-1 number of min agents and 1 max agent
        #Terminal states are either win/loss states or states at the maximum depth
        #Use self.evaluationFunction to evaluate terminal states
        #Return the action that leads to the best value for the max agent at the root node
        
        bestValue = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            if bestValue < self.__minimax__(successor, self.depth, 1):
                bestValue = self.__minimax__(successor, self.depth, 1)
                bestAction = action
         
        return bestAction
        
    def __minimax__(self,gameState, depth, agentIndex):
        """
        Returns the minimax value of the gameState
        """
        value = 0
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()
        if depth == 0:
            return self.evaluationFunction(gameState)
        
        if agentIndex == 0: #Max agent
            value = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = max(value, self.minimax(successor, depth, 1))
            return value
        else: #Min agent
            value = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    value = min(value, self.minimax(successor, depth - 1, 0))
                else:
                    value = min(value, self.minimax(successor, depth, agentIndex + 1))
        return value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        #Basicallty the same as minimax but with alpha-beta pruning
         
        bestValue = float('-inf')
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = self.__alphabeta__(successor, self.depth, 1, alpha, beta)
            if bestValue < value:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)
         
        return bestAction
        
    def __alphabeta__(self,gameState, depth, agentIndex, alpha, beta):
        """
        Returns the minimax value of the gameState using alpha-beta pruning
        """
        value = 0
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()
        if depth == 0:
            return self.evaluationFunction(gameState)
        
        if agentIndex == 0: #Max agent
            return self.__max_value__(gameState, value, depth ,alpha, beta)
        else: #Min agent
            return self.__min_value__(gameState, value, depth, agentIndex, alpha, beta)
    
    def __max_value__(self, gameState, value, depth, alpha, beta):
        """
        Returns the max value of the gameState
        """
        actions = gameState.getLegalActions(0)
        if not actions:
            return self.evaluationFunction(gameState)
        value = float('-inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = max(value, self.alphabeta(successor, depth, 1, alpha, beta))
            if value > beta:
                return value
            alpha = max(alpha, value)
        return value
    
    def __min_value__(self, gameState, value, depth, agentIndex, alpha, beta):
        """
        Returns the min value of the gameState
        """
        actions = gameState.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(gameState)
        value = float('inf')
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                value = min(value, self.alphabeta(successor, depth - 1, 0, alpha, beta))
            else:
                value = min(value, self.alphabeta(successor, depth, agentIndex + 1,alpha, beta))
            if value < alpha:
                return value
            beta = min(beta, value)
        return value
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
        #Basically the same as minimax but with expectimax instead of min nodes
        #Use the average value of the successor states for the expectimax nodes
        bestValue = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = self.__expectimax__(successor, self.depth, 1)
            if bestValue < value:
                bestValue = value
                bestAction = action
        return bestAction
    def __expectimax__(self,gameState, depth, agentIndex):
        """
        Returns the expectimax value of the gameState
        """
        value = 0
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()
        if depth == 0:
            return self.evaluationFunction(gameState)
        
        if agentIndex == 0: #Max agent
            value = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = max(value, self.__expectimax__(successor, depth, 1))
            return value
        else: #Expectimax agent
            value = 0
            actions = gameState.getLegalActions(agentIndex)
            prob = 1.0 / len(actions) if actions else 0
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    value += prob * self.__expectimax__(successor, depth - 1, 0)
                else:
                    value += prob * self.__expectimax__(successor, depth, agentIndex + 1)
        return value

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: <write something here so we know what you did>
    """
    #Talk to Tony about different algorithms to consider
    #Ideas: 
    #Value iteration or Policy iteration is the first that come to mind, but this isn't really a MDP
    #Maybe A* search if we can come up with a good heuristic

    return util.raiseNotDefined
    

    

# Abbreviation
better = betterEvaluationFunction
