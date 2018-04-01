import numpy as np

from experiments.problems.multiObjective.sequence import Sequence

from src.boundaryConstraints.vectorEvaluatedPSO import VectorEvaluatedPSO
from mlpy.numberGenerator.bounds import Bounds

ITERATIONS = 2000
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

pso = VectorEvaluatedPSO()

pso.error = sequence
pso.bounds = Bounds(0, sequence.maxlen + 1)
pso.initialPosition = Bounds(0, 3)

pso.num_particles = NUM_PARTICLES
pso.num_dimensions = sequence.size
pso.weight = INERTIA_WEIGHT
pso.cognitiveConstant = COGNITIVE_CONSTANT
pso.socialConstant = SOCIAL_CONSTANT

trainingErrors, trainingError = pso.train(ITERATIONS, ITERATIONS_SAMPLE_SIZE)

sequence.getFitness(pso.group_best_position)
print("Swarm1 - Result: ")
sequence.printFinalAlignemnt()

sequence.getFitness(pso.group_best_position2)
print("Swarm2 - Result: ")
sequence.printFinalAlignemnt()


