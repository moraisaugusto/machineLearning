import random
import sys
import sys
import numpy as np
from math import *


class EvolutionaryAlgorithm:
    """ Evolutionary Algorithm for Alpine Function"""

    max_i = 10
    min_i = -10
    precision = 8
    ps_Mutation= 0 
    n_Mutation = 25
    success_Mutation = 0
    alpha = 0.005 # 0.005 is good for deterministic mutation
    lastIteration = 1
    mutationKind = None 

    def __init__(self, popSize, generations, pc, pm, mutationKind):
        self.popSize = popSize
        self.generations = generations
        self.pc= pc
        self.pm = pm
        self.mutationKind = mutationKind


    def mutationAutoAdaptive(self, cromo, currentIteration):
        tal = 0.005 
        self.alpha = self.alpha * exp(np.random.normal(0, tal)) 

        for i in range(2):

            mutationRate = np.random.normal(0, self.alpha)
            cromo[i][0] = round(cromo[i][0] + mutationRate, self.precision)
            cromo[i][1] = round(cromo[i][1] + mutationRate, self.precision)

        return cromo

    def mutationFeedback(self, cromo, currentIteration):

        c = 0.85

        # Only update alpha if current Iteration is multiple by n mutation (step)
        if (currentIteration % self.n_Mutation == 0) and (self.lastIteration != currentIteration):
            self.ps_Mutation = self.success_Mutation / (self.n_Mutation * self.popSize)

            if self.ps_Mutation > 0.20:
                self.alpha = ( self.alpha / c ) # same as Eiben Book
            elif self.ps_Mutation < 0.20:
                self.alpha = ( self.alpha * c ) # same as Eiben Book
            else:
                self.alpha = self.alpha # same as Eiben Book

            # reseting successful rate
            self.success_Mutation = 0

            self.lastIteration = currentIteration

        # performing  mutation
        for i in range(2):

            oldFitness = self.evalFitness(cromo[i])

            mutationRate = np.random.normal(0, self.alpha)
            cromo[i][0] = round(cromo[i][0] + mutationRate, self.precision)
            cromo[i][1] = round(cromo[i][1] + mutationRate, self.precision)

            newFitness = self.evalFitness(cromo[i])

            if oldFitness < newFitness:
                self.success_Mutation += 1


        return cromo

    def mutationDeterministic(self, cromo, currentIteration):

    # let's do mutation on X coords
        for i in range(2):
            if random.random() < self.pm:
                alpha = 1 - 0.9 * (currentIteration/self.generations)
                cromo[i][0] = round(cromo[i][0] * alpha, self.precision)
                cromo[i][1] = round(cromo[i][1] * alpha, self.precision)

        return cromo

    def rouletteWheel(self, currentPop):
        popFitness = list()
        rouletteWheel = list()
        allPopFitness = float()

        # getting max solution fitness
        maxSolution = self.worstSolution(currentPop)

        # maximizing problem for roulete wheel 
        for i in range(self.popSize):
            currentFitness = maxSolution - self.evalFitness(currentPop[i])
            popFitness.append(currentFitness)

        rouletteWheel = list()
        propFitness = list()

        # If the sum of all pop Fitness is 0.0 then I set it to 0.0001 (to not get an error)
        allPopFitness = sum(popFitness)
        if allPopFitness == 0.0:
            allPopFitness = 0.0001

        # Maximizing fitness to create roulette Wheel
        for i in range(self.popSize):
            aux = popFitness[i] / allPopFitness
            if aux == 0:
                aux = 0.0000000000001
            rouletteWheel.append(aux)

        # Getting the winner
        aux = rouletteWheel[0] 
        raffledNumber = random.uniform(0, sum(rouletteWheel))  #sum(rouletteWheel) is +- 1.0

        for i in range(0, self.popSize):

            if raffledNumber <= aux:
                # This guy is the WINNER!!!!
                return currentPop[i]
                break

            aux += rouletteWheel[i+1]

    def doCrossover(self, parentOne, parentTwo):

        # Getting Random alpha 
        alpha = round(random.uniform(0.1,0.9),1)

        # Offsprings
        offsprings = list()

        # Doing crossover
        parentOne[0] = round(parentOne[0] , self.precision)
        parentOne[1] = round(parentOne[1] , self.precision)
        parentTwo[0] = round(parentTwo[0] ,self.precision)
        parentTwo[1] = round(parentTwo[1] ,self.precision)

        offsprings.append(parentOne)
        offsprings.append(parentTwo)

        return offsprings


    def createPop(self):
        pop = list()
        for i in range(self.popSize):
            cromo = list()
            x = round(random.uniform(self.min_i, self.max_i),self.precision)
            y = round(random.uniform(self.min_i, self.max_i),self.precision)
            cromo.append(x)
            cromo.append(y)

            # only for autoadaptive Mutation
            if self.mutationKind == 3:
                alpha = round(random.uniform(0,1), 2)
                cromo.append(alpha)

            pop.append(cromo)

        return pop

    def evalFitness(self, coords):

        # Eval Fitness
        fAlpine = float()
        for i in range(2):
            xRadians = radians(coords[i])
            sinX = float(sin(xRadians))
            fAlpine += abs( coords[i] * sinX + 0.1 * coords[i] )

        return round(fAlpine, self.precision)


    def worstSolution(self, currentPop):

        # Evaluate Fitness
        worstFitness = self.evalFitness(currentPop[0])
        worstCoords = currentPop[0]

        for i in range(1, self.popSize):
            currentFitness = self.evalFitness(currentPop[i])

            if worstFitness < currentFitness:
                worstFitness = currentFitness
                worstCoords = currentPop[i]

        return worstFitness


    def averageSolution(self, currentPop):

        currentFitness = list()
        for i in range(self.popSize):
            currentFitness.append(self.evalFitness(currentPop[i]))

        return round( sum(currentFitness) / self.popSize, self.precision )


    def bestSolution(self, currentPop):

        # Evaluate Fitness
        bestFitness = self.evalFitness(currentPop[0])
        bestCoords = currentPop[0]

        for i in range(1, self.popSize):
            currentFitness = self.evalFitness(currentPop[i])

            if bestFitness > currentFitness:
                bestFitness = currentFitness
                bestCoords = currentPop[i]

        return bestFitness

