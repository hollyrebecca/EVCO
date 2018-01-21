# This code defines the agent (as in the playable version) in a way that can be called and 
# executed from an evolutionary algorithm. The code is partial and will not execute. You need to 
# add to the code to create an evolutionary algorithm that evolves and executes a snake agent.
import curses
import random
import operator
import math
from functools import partial

import numpy
import pygraphviz as pgv
import matplotlib.pyplot as plt
import multiprocessing

from deap import algorithms 
from deap import base
from deap import creator
from deap import tools
from deap import gp

S_RIGHT, S_LEFT, S_UP, S_DOWN = 0,1,2,3
XSIZE,YSIZE = 14,14
GRIDSIZE = YSIZE*XSIZE
NFOOD = 1 # TODO: CHECK THAT THERE ARE ENOUGH SPACES LEFT FOR THE FOOD (IF THE TAIL IS VERY LONG)
TOTALFOOD = 185 # total possible amount of food that can be eaten 
NGEN = 500 # number of generations
NPOP = 1000 # size of the population
maxDepth = 17 # depth of the decision tree
CXPB = 0.6 # probability of mating
MUTX = 0.5 # probability of mutation
NCOUNT = 1

def progn(*args):
    for arg in args:
        arg()

def prog2(out1, out2): 
    return partial(progn,out1,out2)

def prog3(out1, out2, out3):     
    return partial(progn,out1,out2,out3)

def if_then_else(condition, out1, out2):
    out1() if condition() else out2()

# This class can be used to create a basic player object (snake agent)
class SnakePlayer(list):
	global S_RIGHT, S_LEFT, S_UP, S_DOWN
	global XSIZE, YSIZE

	def __init__(self):
		self.direction = S_RIGHT
		self.body = [ [4,10], [4,9], [4,8], [4,7], [4,6], [4,5], 
				[4,4], [4,3], [4,2], [4,1],[4,0] ]
		self.score = 0
		self.ahead = []
		self.food = []

	def _reset(self):
		self.direction = S_RIGHT
		self.body[:] = [ [4,10], [4,9], [4,8], [4,7], [4,6], [4,5], 
					[4,4], [4,3], [4,2], [4,1],[4,0] ]
		self.score = 0
		self.ahead = []
		self.food = []

	def getAheadLocation(self):
		self.ahead = [ self.body[0][0] + (self.direction == S_DOWN and 1) + 
				(self.direction == S_UP and -1), self.body[0][1] + 
				(self.direction == S_LEFT and -1) + 
				(self.direction == S_RIGHT and 1)] 

	def getLeftLocation(self):
		return [ self.body[0][0] + (self.direction == S_RIGHT and -1) + 
				(self.direction == S_LEFT and 1), self.body[0][1] + 
				(self.direction == S_DOWN and 1) + 
				(self.direction == S_UP and -1)] 

	def getRightLocation(self):
		return [ self.body[0][0] + (self.direction == S_LEFT and -1) + 
				(self.direction == S_RIGHT and 1), self.body[0][1] + 
				(self.direction == S_DOWN and -1) + 
				(self.direction == S_UP and 1)] 

	def getAhead2Location(self):
		self.getAheadLocation()
		ahead = self.ahead
		ahead2 = [ ahead[0] + (self.direction == S_DOWN and 1) + 
				(self.direction == S_UP and -1), ahead[1] + 
				(self.direction == S_LEFT and -1) + 
				(self.direction == S_RIGHT and 1)] 
		return ahead2

	def updatePosition(self):
		self.getAheadLocation()
		self.body.insert(0, self.ahead)

	## You are free to define more sensing options to the snake

	def goUp(self):
		self.direction = S_UP

	def goRight(self):
		self.direction = S_RIGHT

	def goDown(self):
		self.direction = S_DOWN

	def goLeft(self):
		self.direction = S_LEFT

	def stayStraight(self):
		pass

	def isMovingUp(self):
		return self.direction == S_UP

	def isMovingRight(self):
		return self.direction == S_RIGHT

	def isMovingDown(self):
		return self.direction == S_DOWN

	def isMovingLeft(self):
		return self.direction == S_LEFT
	
	def ifMovingUp(self, out1, out2):
		return partial(if_then_else, self.isMovingUp, out1, out2)

	def ifMovingRight(self, out1, out2):
		return partial(if_then_else, self.isMovingRight, out1, out2)

	def ifMovingDown(self, out1, out2):
		return partial(if_then_else, self.isMovingDown, out1, out2)

	def ifMovingLeft(self, out1, out2):
		return partial(if_then_else, self.isMovingLeft, out1, out2)

	def snakeHasCollided(self):
		self.hit = False
		if self.body[0][0] == 0 or self.body[0][0] == (YSIZE-1) or \
			self.body[0][1] == 0 or self.body[0][1] == (XSIZE-1): 
			self.hit = True
		if self.body[0] in self.body[1:]: 
			self.hit = True
		return self.hit

	def senseWallAhead(self):
		self.getAheadLocation()
		return (self.ahead[0] == 0 or self.ahead[0] == (YSIZE-1) or \
			self.ahead[1] == 0 or self.ahead[1] == (XSIZE-1))

	def senseWallRight(self):
		ahead = self.getRightLocation()
		return (ahead[0] == 0 or ahead[0] == (YSIZE-1) or \
			ahead[1] == 0 or ahead[1] == (XSIZE-1))

	def senseWallLeft(self):
		ahead = self.getLeftLocation()
		return (ahead[0] == 0 or ahead[0] == (YSIZE-1) or \
			ahead[1] == 0 or ahead[1] == (XSIZE-1))

	def senseWall2Ahead(self):
		ahead = self.getAhead2Location()
		return (ahead[0] == 0 or ahead[0] == (YSIZE-1) or \
			ahead[1] == 0 or ahead[1] == (XSIZE-1))

	def senseFoodAhead(self):
		self.getAheadLocation()
		return self.ahead in self.food

	def senseFoodAllAhead(self):
		self.getAheadLocation()
		if self.direction == S_UP:
			if self.food[0][1] < self.ahead[1]:
				return True
		elif self.direction == S_DOWN:
			if self.food[0][1] > self.ahead[1]:
				return True
		elif self.direction == S_RIGHT:
			if self.food[0][0] > self.ahead[0]:
				return True		
		elif self.direction == S_LEFT:
			if self.food[0][0] < self.ahead[0]:
				return True
		return False

	def ifFoodAllAhead(self, out1, out2):
		return partial(if_then_else, self.senseFoodAllAhead, out1, out2)

	# use direction to determine where the food is in terms of the whole board
	def senseFoodUp(self):
		if not self.food:
			return None
		foodloc = self.food[0][0]
		loc = foodloc - self.body[0][0]
		if loc < 1:
			return False
	
	# use direction to determine where the food is in terms of the whole board
	def senseFoodRight(self):
		if not self.food:
			return None
		foodloc = self.food[0][1]
		loc = foodloc - self.body[0][1]
		if loc < 1:
			return False	

	# use direction to determine where the food is in terms of the whole board
	def senseFoodDown(self):
		if not self.food:
			return None
		foodloc = self.food[0][0]
		loc = foodloc - self.body[0][0]
		if loc > 1:
			return False
	
	# use direction to determine where the food is in terms of the whole board
	def senseFoodLeft(self):
		if not self.food:
			return None
		foodloc = self.food[0][1]
		loc = foodloc - self.body[0][1]
		if loc > 1:
			return False	
	
	def senseTailAhead(self):
		self.getAheadLocation()
		return self.ahead in self.body

	def senseTailRight(self):
		self.getRightLocation()
		return self.ahead in self.body

	def senseTailLeft(self):
		self.getLeftLocation()
		return self.ahead in self.body

	def senseTail2Ahead(self):
		return self.getAhead2Location() in self.body

	def senseDangerAhead(self):
		return self.senseTailAhead() or self.senseWallAhead()

	def senseDangerRight(self):
		return self.senseTailRight() or self.senseWallRight()

	def senseDangerLeft(self):
		return self.senseTailLeft() or self.senseWallLeft()

	def senseDanger2Ahead(self):
		return self.senseTail2Ahead() or self.senseWall2Ahead()

	def ifWallAhead(self, out1, out2):
		return partial(if_then_else, self.senseWallAhead, out1, out2)

	def ifWall2Ahead(self, out1, out2):
		return partial(if_then_else, self.senseWall2Ahead, out1, out2)

	def ifDangerAhead(self, out1, out2):
		return partial(if_then_else, self.senseDangerAhead, out1, out2)

	def ifDangerRight(self, out1, out2):
		return partial(if_then_else, self.senseDangerRight, out1, out2)

	def ifDangerLeft(self, out1, out2):
		return partial(if_then_else, self.senseDangerLeft, out1, out2)

	def ifDanger2Ahead(self, out1, out2):
		return partial(if_then_else, self.senseDanger2Ahead, out1, out2)

	def ifFoodUp(self, out1, out2):
		return partial(if_then_else, self.senseFoodUp, out1, out2)

	def ifFoodRight(self, out1, out2):
		return partial(if_then_else, self.senseFoodRight, out1, out2)

	def ifFoodDown(self, out1, out2):
		return partial(if_then_else, self.senseFoodDown, out1, out2)

	def ifFoodLeft(self, out1, out2):
		return partial(if_then_else, self.senseFoodLeft, out1, out2)

# This function places a food item in the environment
def placeFood(snake):
	food = []
	if (GRIDSIZE) == len(snake.body):
		return None
	timer = 0
	while len(food) < NFOOD and timer < GRIDSIZE+1:
		potentialfood = [random.randint(1, (YSIZE-2)), random.randint(1, (XSIZE-2))]
		if not (potentialfood in snake.body) and not (potentialfood in food):
			food.append(potentialfood)
		timer += 1
	if timer == GRIDSIZE:
		return None
	snake.food = food  # let the snake know where the food is
	return(food)


snake = SnakePlayer()

# This outline function is the same as runGame (see below). However,
# it displays the game graphically and thus runs slower
# This function is designed for you to be able to view and assess
# your strategies, rather than use during the course of evolution
def displayStrategyRun(individual):
	global snake
	global pset

	routine = gp.compile(individual, pset)

	curses.initscr()
	win = curses.newwin(YSIZE, XSIZE, 0, 0)
	win.keypad(1)
	curses.noecho()
	curses.curs_set(0)
	win.border(0)
	win.nodelay(1)
	win.timeout(120)

	snake._reset()
	food = placeFood(snake)

	for f in food:
		win.addch(f[0], f[1], '@')

	timer = 0
	collided = False
	while not collided and not timer == ((2*XSIZE) * YSIZE):

		# Set up the display
		win.border(0)
		win.addstr(0, 2, 'Score : ' + str(snake.score) + ' ')
 		win.getch()

		## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
		routine()

		snake.updatePosition()

		if snake.body[0] in food:
			snake.score += 1
			for f in food: win.addch(f[0], f[1], ' ')
			food = placeFood(snake)
			if food == None:
				break
			for f in food: win.addch(f[0], f[1], '@')
			timer = 0
		else:    
			last = snake.body.pop()
			win.addch(last[0], last[1], ' ')
			timer += 1 # timesteps since last eaten
		win.addch(snake.body[0][0], snake.body[0][1], 'o')

		collided = snake.snakeHasCollided()
		hitBounds = (timer == ((2*XSIZE) * YSIZE))

	curses.endwin()

	print collided
	print hitBounds
	raw_input("Press to continue...")

	return snake.score,


# This outline function provides partial code for running the game with an evolved agent
# There is no graphical output, and it runs rapidly, making it ideal for
# you need to modify it for running your agents through the game for evaluation
# which will depend on what type of EA you have used, etc.
# Feel free to make any necessary modifications to this section.
def runGame(individual):
	global snake
	global pset

	routine = gp.compile(individual, pset)

	aggScore = 0
	for x in range(0, NCOUNT):
		totalScore = 0

		snake._reset()
		food = placeFood(snake)
		timer = 0

		while not snake.snakeHasCollided() and not timer == GRIDSIZE:

			#if snake.score == (XSIZE * YSIZE) - snake.initial+1:
			#	break

			## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
			routine()
			#individual()

			snake.updatePosition()

			if snake.body[0] in food:
				snake.score += 1
				food = placeFood(snake)
				if food == None:
					break
				timer = 0
			else:    
				snake.body.pop()
				timer += 1 # timesteps since last eaten
		
			totalScore += snake.score
			snake.score = 0

		#if timer == XSIZE*YSIZE:
		#	return TOTALFOOD + 5,
	#		return -5,

		if totalScore == 0:
			distanceFromFood = 10
			ydist = math.sqrt((int(snake.food[0][0]) - int(snake.body[0][0]))**2)
			xdist = math.sqrt((int(snake.food[0][1]) - int(snake.body[0][1]))**2)

			distanceFromFood = math.ceil(ydist) + math.ceil(xdist)

			return TOTALFOOD + distanceFromFood,
		#	return 0 - distanceFromFood,

		aggScore += (TOTALFOOD - totalScore)

	avgScore = aggScore / NCOUNT

	return avgScore,
	#return totalScore,

def evalRunGame(individual, runs):
	global snake
	global pset

	routine = gp.compile(individual, pset)

	aggScore = 0
	for x in range(0, runs):
		totalScore = 0

		snake._reset()
		food = placeFood(snake)
		timer = 0

		while not snake.snakeHasCollided() and not timer == GRIDSIZE:

			## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
			routine()

			snake.updatePosition()

			if snake.body[0] in food:
				snake.score += 1
				food = placeFood(snake)
				if food == None:
					break
				timer = 0
			else:    
				snake.body.pop()
				timer += 1 # timesteps since last eaten
		
			totalScore += snake.score
			snake.score = 0


		if totalScore == 0:
			distanceFromFood = 10
			ydist = math.sqrt((int(snake.food[0][0]) - int(snake.body[0][0]))**2)
			xdist = math.sqrt((int(snake.food[0][1]) - int(snake.body[0][1]))**2)

			distanceFromFood = math.ceil(ydist) + math.ceil(xdist)

			return TOTALFOOD + distanceFromFood,

		aggScore += (TOTALFOOD - totalScore)

	avgScore = aggScore/runs
	return avgScore,

#TO-DO

#PrimitiveSet definitions
pset = gp.PrimitiveSet("MAIN", 0)
pset.addPrimitive(prog2, 2)
#pset.addPrimitive(prog3, 3)

pset.addPrimitive(snake.ifMovingUp, 2, name="ifMovingUp")
pset.addPrimitive(snake.ifMovingDown, 2, name="ifMovingDown")
pset.addPrimitive(snake.ifMovingLeft, 2, name="ifMovingLeft")
pset.addPrimitive(snake.ifMovingRight, 2, name="ifMovingRight")
pset.addPrimitive(snake.ifDangerAhead, 2, name="ifDangerAhead")
pset.addPrimitive(snake.ifDangerLeft, 2, name="ifDangerLeft")
pset.addPrimitive(snake.ifDangerRight, 2, name="ifDangerRight")
pset.addPrimitive(snake.ifDanger2Ahead, 2, name="ifDanger2Ahead")
#pset.addPrimitive(snake.ifWallAhead, 2, name="ifWallAhead")
#pset.addPrimitive(snake.ifWall2Ahead, 2, name="ifWall2Ahead")
pset.addPrimitive(snake.ifFoodAllAhead, 2, name="ifFoodAllAhead")
pset.addPrimitive(snake.ifFoodUp, 2, name="ifFoodUp")
pset.addPrimitive(snake.ifFoodRight, 2, name="ifFoodRight")
#pset.addPrimitive(snake.ifFoodDown, 2, name="ifFoodDown")
#pset.addPrimitive(snake.ifFoodLeft, 2, name="ifFoodLeft")

pset.addTerminal(snake.goUp, name="goUp")
pset.addTerminal(snake.goDown, name="goDown")
pset.addTerminal(snake.goRight, name="goRight")
pset.addTerminal(snake.goLeft, name="goLeft")
pset.addTerminal(snake.stayStraight, name="stayStraight")
#pset.addTerminal(snake.goUp)
#pset.addTerminal(snake.goDown)
#pset.addTerminal(snake.goRight)
#pset.addTerminal(snake.goLeft)
#pset.addTerminal(snake.stayStraight)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)
#creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6)
#toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=4)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", runGame)
toolbox.register("select", tools.selTournament, tournsize=7)
#toolbox.register("select", tools.selDoubleTournament, fitness_size=5, parsimony_size=1.2, fitness_first=True)
#toolbox.register("mate", gp.cxUniform)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=6)
#toolbox.register("expr_mut", gp.genFull, min_=0, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
#toolbox.register("mutate", gp.mutGaussian, mu=0, sigma=0.4, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

toolbox.register("compile", gp.compile, pset=pset)

def plotGraph(logbook):
	gen = logbook.select("gen")
	fit_mins = logbook.chapters["fitness"].select("min")
	size_avgs = logbook.chapters["size"].select("avg")

	fig, ax1 = plt.subplots()
	line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
	ax1.set_xlabel("Generation")
	ax1.set_ylabel("Fitness", color="b")
	for tl in ax1.get_yticklabels():
	    tl.set_color("b")

	ax2 = ax1.twinx()
	line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
	ax2.set_ylabel("Size", color="r")
	for tl in ax2.get_yticklabels():
	    tl.set_color("r")

	lns = line1 + line2
	labs = [l.get_label() for l in lns]
	ax1.legend(lns, labs, loc="center right")

	fig.savefig("/usr/userfs/h/hrh517/Downloads/plot.png")

def main():
	global snake
	global pset

	#random.seed(128)

	pool = multiprocessing.Pool()
	toolbox.register("map", pool.map)

	## THIS IS WHERE YOUR CORE EVOLUTIONARY ALGORITHM WILL GO #
	pop = toolbox.population(n=NPOP)
	hof = tools.HallOfFame(1)

	stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
	stats_size = tools.Statistics(len)
	mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
	mstats.register("avg", numpy.mean)
	mstats.register("std", numpy.std)
	mstats.register("min", numpy.min)
	mstats.register("max", numpy.max)

	try:
		pop, log = algorithms.eaSimple(pop, toolbox, CXPB, MUTX, 
						NGEN, stats=mstats, halloffame=hof, verbose=True)

		best = tools.selBest(pop, 1)[0]

		#evalRuns = 5
		#evalRunGame(best, evalRuns)

		# display the run of the best individual	
		#displayStrategyRun(best)

	except KeyboardInterrupt:
		pool.terminate()
		pool.join()
		raise KeyboardInterrupt

	#plotGraph(logbook)

	# section for creating graph to represent the evolution
	#expr = toolbox.individual()
	nodes, edges, labels = gp.graph(best)
	g = pgv.AGraph()
	g.add_nodes_from(nodes)
	g.add_edges_from(edges)
	g.layout(prog="dot")

	for i in nodes:
		n = g.get_node(i)
		n.attr["label"]=labels[i]

	g.draw("tree311.pdf")


	return mstats.compile(pop)



if __name__ == "__main__":
	for i in range(0, 30):
		out = main()
		run = out
		row = (run['fitness']['avg'], run['fitness']['min'], run['fitness']['std'], run['size']['avg'], run['size']['max'], run['size']['std'], "\r")
		runFile = open('cxOnePoint.csv', 'a+')
		runFile.write(",".join(map(str,row)))
		runFile.close()

