import numpy as np

from experiments.problems.multiObjective.sequence import Sequence

from src.boundaryConstraints.dynamicVEPSO import DynamicVEPSO
from mlpy.numberGenerator.bounds import Bounds

ITERATIONS = 2000
ITERATIONS_SAMPLE_SIZE = 10
SAMPLES = 1

NUM_PARTICLES = 50
INERTIA_WEIGHT = 0.7
COGNITIVE_CONSTANT = SOCIAL_CONSTANT = 1.4

sequence = Sequence([['a', 'b', 'c', 'd', 'e', 'f'],
                     ['b', 'b', 'd', 'h', 'g'],
                     ['c', 'a', 'b', 'f']])

sequence2 = Sequence([['a', 'b', 'c', 'd', 'e', 'g'],
                     ['a', 'b', 'd', 'h', 'g'],
                     ['c', 'a', 'b', 'h']])

sequence3 = Sequence([['a', 'b', 'c', 'd', 'e', 'f'],
                     ['g', 'h', 'i', 'h', 'g'],
                     ['c', 'a', 'b', 'a']])

thing1, valid = sequence.getFitness(np.zeros(sequence.size))
print("Original: ")
sequence.printFinalAlignemnt()

pso = DynamicVEPSO()

pso.error = sequence2
pso.bounds = Bounds(0, sequence.maxlen + 1)
pso.initialPosition = Bounds(0, sequence.maxlen + 1)

pso.num_particles = NUM_PARTICLES
pso.num_dimensions = sequence.size
pso.weight = INERTIA_WEIGHT
pso.cognitiveConstant = COGNITIVE_CONSTANT
pso.socialConstant = SOCIAL_CONSTANT

trainingErrors, trainingError = pso.train(ITERATIONS, ITERATIONS_SAMPLE_SIZE)

thing2, valid = sequence.getFitness(pso.group_best_position)
print("Result1: ", valid)
sequence.printFinalAlignemnt()

thing3, valid = sequence.getFitness(pso.group_best_position2)
print("Result2: ", valid)
sequence.printFinalAlignemnt()

print(thing1, thing2)
print(sequence.dominant(thing1, thing2))


