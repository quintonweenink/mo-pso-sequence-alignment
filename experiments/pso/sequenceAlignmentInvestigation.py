import numpy as np

from experiments.problems.multiObjective.sequence import Sequence

from src.boundaryConstraints.boundaryConstraintPSO import BoundaryConstraintPSO
from mlpy.numberGenerator.bounds import Bounds

ITERATIONS = 500
ITERATIONS_SAMPLE_SIZE = 10
SAMPLES = 1

NUM_PARTICLES = 30
INERTIA_WEIGHT = 0.7
COGNITIVE_CONSTANT = SOCIAL_CONSTANT = 1.4

sequence = Sequence([['a', 'b', 'c', 'd', 'e', 'f'],
                     ['b', 'b', 'd', 'h', 'g'],
                     ['c', 'a', 'b', 'f']])
sequence.getFitness(np.zeros(sequence.size))
print("Original: ")
sequence.printFinalAlignemnt()

pso = BoundaryConstraintPSO()

def fitness(position):
    error, valid = sequence.getFitness(position)
    return (error[0] * sequence.maxlen) + error[1]


from boundaryConstraints import *

pso.boundaryConstraint = bc2

pso.error = fitness
pso.bounds = Bounds(0, sequence.maxlen + 1)
pso.initialPosition = Bounds(0, sequence.maxlen + 1)

pso.num_particles = NUM_PARTICLES
pso.num_dimensions = sequence.size
pso.weight = INERTIA_WEIGHT
pso.cognitiveConstant = COGNITIVE_CONSTANT
pso.socialConstant = SOCIAL_CONSTANT

trainingErrors, trainingError = pso.train(ITERATIONS, ITERATIONS_SAMPLE_SIZE)

sequence.getFitness(pso.group_best_position)
print("Result: ")
sequence.printFinalAlignemnt()


