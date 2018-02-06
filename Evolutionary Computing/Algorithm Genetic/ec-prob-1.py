#!/usr/bin/env python3

# Do the import
import matplotlib.pyplot as plt
from libs.population import *
import random
import sys


ag = EvolutionaryAlgorithm(100, 100, 0.7, 0.125)

# Create Population
currentPop = ag.createPop()

halfPopSize = int ( ag.popSize / 2 )

allBestFitness = list()
allWorstFitness = list()
allAverageFitness = list()

bestCurrentFitness = ag.bestSolution(currentPop)
allBestFitness.append(bestCurrentFitness) 
#bestCurrentFitness= 9999 

allAverageFitness.append(ag.averageSolution(currentPop) )

worstCurrentFitness= ag.worstSolution(currentPop)
allWorstFitness.append(worstCurrentFitness) 
#worstFitnessGeneration = 0
#print(bestCurrentFitness)
#print(worstCurrentFitness)
#sys.exit()
newPop = list()

for j in range(ag.generations):
    for i in range( halfPopSize ):
        sortedParentOne = ag.rouletteWheel(currentPop)
        sortedParentTwo = ag.rouletteWheel(currentPop)


        if sortedParentOne is None or sortedParentTwo is None:
            print("deu merda")

        # Will do Crossover?
        if random.random() <= ag.pc:
            offsprings = list()

            offsprings = ag.doCrossover(sortedParentOne, sortedParentTwo)

            # Let's do Mutation (if its necessary...)
            offsprings = ag.mutation(offsprings, j)

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
            offsprings = ag.mutation(offsprings, j)

            # Adding Parents on new Population
            newPop.append(offsprings[0])
            newPop.append(offsprings[1])

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
with plt.style.context('fivethirtyeight'):
        plt.plot(allBestFitness, label="Melhores")
        plt.plot(allAverageFitness, label="Médias")
        plt.plot(allWorstFitness, label="Piores")
        plt.grid(True)
        plt.axis([0, ag.generations, 0, 10])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=5, borderaxespad=0.)
        plt.title('Evolução do AG', bbox={'facecolor': '0.8', 'pad': 5})

#plt.show()
filename = "teste-OK-PC-{:f}-PM-{:f}.svg".format(ag.pc, ag.pm)

plt.savefig(filename, format="svg")
plt.close()
#plt.save("teste", ext="png", close=True, verbose=True)
