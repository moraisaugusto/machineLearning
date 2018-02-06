#!/usr/bin/env python3

# Do the import
import matplotlib.pyplot as plt
from libs.population import *
import random
import sys

# initial vars
allBestFitness = list()
allWorstFitness = list()
allAverageFitness = list()

# penult parameter is only use on deterministic mutation
# 1: deterministic | 2: feedback | 3: autoadaptive
ag = EvolutionaryAlgorithm(100, 100, 0.7, 0.125, 3)

# Create Population
currentPop = ag.createPop()

# getting Best, Worst and average fitness
bestCurrentFitness = ag.bestSolution(currentPop)
allBestFitness.append(bestCurrentFitness)
allAverageFitness.append(ag.averageSolution(currentPop) )
worstCurrentFitness= ag.worstSolution(currentPop)
allWorstFitness.append(worstCurrentFitness)

newPop = list()
halfPopSize = int ( ag.popSize / 2 )


# Performing all Algorith, Genetic
for j in range(ag.generations):
    for i in range( halfPopSize ):
        sortedParentOne = ag.rouletteWheel(currentPop)
        sortedParentTwo = ag.rouletteWheel(currentPop)

        # Will do Crossover?
        if random.uniform(0,1) <= ag.pc:
            offsprings = list()

            offsprings = ag.doCrossover(sortedParentOne, sortedParentTwo)

            # Let's do Mutation (if its necessary...)
            offsprings = ag.mutationAutoAdaptive(offsprings, j+1)
            #offsprings = ag.mutationFeedback(offsprings, j+1)
            #offsprings = ag.mutationDeterministic(offsprings, j+1)

            # adding offsprings on new Population
            newPop.append(offsprings[0])
            newPop.append(offsprings[1])

        # Let's get the parents
        else:
            offsprings = list()

            # Parents copied to offsprings
            offsprings.append(sortedParentOne)
            offsprings.append(sortedParentTwo)

            # Let's do Mutation (if its necessary...)
            offsprings = ag.mutationAutoAdaptive(offsprings, j+1)
            #offsprings = ag.mutationFeedback(offsprings, j+1)
            #offsprings = ag.mutationDeterministic(offsprings, j+1)

            # Adding Parents on new Population
            newPop.append(offsprings[0])
            newPop.append(offsprings[1])

    # update success tax of Mutation
    if (j+1) % ag.n_Mutation == 0:
        ag.ps_Mutation= ag.success_Mutation / ag.popSize

    # Best Fitness of each Generation
    bestCurrentFitness = ag.bestSolution(newPop)
    allBestFitness.append(bestCurrentFitness)

    # Average Fitness of each Generation
    allAverageFitness.append(ag.averageSolution(newPop) )

    # Worst Fitness of each Generation
    worstCurrentFitness= ag.worstSolution(newPop)
    allWorstFitness.append(worstCurrentFitness)

    # cleaning my new Population
    currentPop = newPop
    newPop = list()


# Ploting Graph
filename = "teste-OK-PC-{:f}-PM-{:f}.svg".format(ag.pc, ag.pm)

with plt.style.context('ggplot'):
        plt.plot(allBestFitness, label="Melhores")
        plt.plot(allAverageFitness, label="Médias")
        plt.plot(allWorstFitness, label="Piores")
        plt.axis([0, ag.generations, 0, 100])
        plt.axis([0, ag.generations, 0, 5])
        plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left", borderaxespad=1)
        plt.ylabel("Fitness")
        plt.xlabel("Gerações")
        #plt.title('Evolução do AG', bbox={'facecolor': '0.8', 'pad': 5})
        plt.savefig(filename, format="svg", bbox_inches='tight')
        plt.close()

print("Best Solution: ", ag.bestSolution(currentPop) )
