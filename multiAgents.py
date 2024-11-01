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


        "*** YOUR CODE HERE ***"
        
        score = 0
        currentFood = currentGameState.getFood().asList()
        capsules = currentGameState.getCapsules()

        for i in range(len(newGhostStates)):
            dist = manhattanDistance(newGhostStates[i].getPosition(),newPos)

            if newPos in currentFood: score+=1
            if  dist<= newScaredTimes[i]: score+=dist*2
            if dist < 3: score -=3
            if newPos in capsules: score+=3
            if newScaredTimes[i] > 0 and dist <= newScaredTimes[i]: score += 5
            if currentGameState.hasWall(newPos[0], newPos[1]): score-=100
            
            distanceToFood = [manhattanDistance(newPos, (xx, yy)) for xx, yy in currentFood]
            score -= min(distanceToFood)/15

        return score

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
        moves = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, x) for x in moves]
        mx = -float('inf')
        goal = 0

        for i in range(len(successors)):
            action = self.minmax(successors[i], 1, 0)
            if action > mx:
                mx = action
                goal = i
        return moves[goal]

    def minmax(self, state, agent, depth):
        if state.isWin() or state.isLose() or self.depth == depth:
            return self.evaluationFunction(state)

        moves = state.getLegalActions(agent)
        successors = [state.generateSuccessor(agent, x) for x in moves]

        if agent == 0:
            return max(self.minmax(successors[i], 1, depth) for i in range(len(successors)))

        if agent + 1 == state.getNumAgents():
            return min(self.minmax(successors[i], 0, depth + 1) for i in range(len(successors)))
        else:
            return min(self.minmax(successors[i], agent+1, depth) for i in range(len(successors)))



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = -float('inf')
        beta = float('inf')

        successors = [state.generateSuccessor(0, x) for x in state.getLegalActions(0)]
        maxx = -float('inf')
        goal = 0

        for x in range(len(successors)):
            action = self.alphabeta(successors[x], 1, 0, alpha, beta)
            if action > maxx:
                maxx = action
                goal = x
                alpha = max(alpha,action)
    
        return state.getLegalActions(0)[goal]

    def alpha(self, state, agent, depth, alpha, beta):
        maxx = -float('inf')
        for x in state.getLegalActions(agent):
            maxx = max(maxx, self.alphabeta(state.generateSuccessor(agent, x),1, depth, alpha, beta))
            if maxx > beta:
                return maxx
            alpha = max(alpha, maxx)
        return maxx

    def beta(self, state, agent, depth, alpha, beta):
        maxx = float('inf')
        for x in state.getLegalActions(agent):
            
            if agent + 1 == state.getNumAgents():
                maxx = min(maxx, self.alphabeta(state.generateSuccessor(agent, x),0, depth + 1, alpha, beta))
            else:
                maxx = min(maxx, self.alphabeta(state.generateSuccessor(agent, x),agent+1 , depth, alpha, beta))
            if maxx < alpha:
                return maxx
            beta = min(beta, maxx)

        return maxx

    def alphabeta(self, state, agent, depth, alpha, beta):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
    
        if agent == 0:
            return self.alpha(state, agent, depth, alpha, beta)
        else:
            return self.beta(state, agent, depth, alpha, beta)



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
        moves = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, x) for x in moves]
        mx = -float('inf')
        goal = 0

        for i in range(len(successors)):
            action = self.expectiMax(successors[i], 1, 0)
            if action > mx:
                mx = action
                goal = i
        return moves[goal]
    
    def expectiMax(self, state, agent, depth):
        if state.isWin() or state.isLose() or self.depth == depth:
            return self.evaluationFunction(state)

        moves = state.getLegalActions(agent)
        successors = [state.generateSuccessor(agent, x) for x in moves]

        if agent == 0:
            return max(self.expectiMax(successors[i], 1, depth) for i in range(len(successors)))


        if agent + 1 == state.getNumAgents():
            # If the next agent is the last ghost, go to the max depth
            return sum(self.expectiMax(successors[i], 0, depth + 1) for i in range(len(successors))) / len(successors)
        else:
            # If there are more ghosts, continue with the next ghost
            return sum(self.expectiMax(successors[i], agent + 1, depth) for i in range(len(successors))) / len(successors)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    whereami = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghosts = currentGameState.getGhostStates()
    capsule = currentGameState.getCapsules()

    # Calculate distances and scores that depend on those thigns for all the like variables ie food, capsule, etc
    foodDistances = [manhattanDistance(whereami, food) for food in foodGrid.asList()]
    if foodDistances:
        closestFood = min(foodDistances)
    else: closestFood = 0

    ghostDist = [manhattanDistance(whereami, ghost.getPosition()) for ghost in ghosts]
    if ghostDist:
        closestGhost = min(ghostDist) 
    else: closestGhost = 0

    capsuleDist = [manhattanDistance(whereami, capsule) for capsule in capsule]
    if capsuleDist:
        closestCapsule = min(capsuleDist) 
    else: closestCapsule = 0

    # final score considering all of these
    score = currentGameState.getScore()
    score += (closestFood + 1)**-1
    score -= 3 * closestGhost
    score -= 3 * closestCapsule

    blocks = currentGameState.getWalls()
    blockDist = [manhattanDistance(whereami,wall) for wall in blocks.asList()]
    if blockDist:
        closestWall = min(blockDist)
    else: closestWall = 0

    score -= closestWall

    return score

# Abbreviation
better = betterEvaluationFunction
