import numpy as np
import matplotlib.pyplot as plt

from experiments.problems.multiObjective.sequence import Sequence

from src.VEPSO import VEPSO
from mlpy.numberGenerator.bounds import Bounds

ITERATIONS = 2000
ITERATIONS_SAMPLE_SIZE = 10
SAMPLES = 30

NUM_PARTICLES = 50
INERTIA_WEIGHT = 0.7
COGNITIVE_CONSTANT = SOCIAL_CONSTANT = 1.4

sequence = Sequence([['a', 'b', 'c', 'd', 'e', 'f'],
                     ['b', 'b', 'd', 'h', 'g'],
                     ['c', 'a', 'b', 'f']])

# sequence = Sequence([['a', 'b', 'c', 'd', 'e', 'f', 'a', 'a', 'h'],
#                      ['f', 'b', 'b', 'd', 'h', 'g', 'h', 'h'],
#                      ['c', 'a', 'b', 'f', 'f', 'e', 'a']])

# sequence = Sequence([['a', 'b', 'c', 'd', 'e', 'f'],
#                      ['g', 'h', 'i', 'j', 'k'],
#                      ['a', 'b', 'c', 'd']])

objectives, valid = sequence.getFitness(np.zeros(sequence.size))
print("Original (valid =", valid, ") objectives =", objectives)
sequence.printFinalAlignemnt()

pso_errors = []
pso_error = []

pso_errors2 = []
pso_error2 = []

pso_errors_mean = []
pso_errors_mean2 = []

for i in range(SAMPLES):
    print(i + 1)
    pso = VEPSO()

    pso.error = sequence
    pso.bounds = Bounds(0, sequence.maxlen + 1)
    pso.initialPosition = Bounds(0, sequence.maxlen + 1)

    pso.num_particles = NUM_PARTICLES
    pso.num_dimensions = sequence.size
    pso.weight = INERTIA_WEIGHT
    pso.cognitiveConstant = COGNITIVE_CONSTANT
    pso.socialConstant = SOCIAL_CONSTANT

    trainingErrors, trainingErrors2, trainingError, trainingError2 = pso.train(ITERATIONS, ITERATIONS_SAMPLE_SIZE)

    pso_errors.append(trainingErrors)
    pso_errors2.append(trainingErrors2)
    pso_error.append(trainingError)
    pso_error2.append(trainingError2)

    objectives, valid = sequence.getFitness(pso.group_best_position)
    print("Result Swarm-1 (valid =", valid, ") objectives =", objectives)
    sequence.printFinalAlignemnt()

    objectives, valid = sequence.getFitness(pso.group_best_position2)
    print("Result Swarm-2: (valid =", valid, ") objectives =", objectives)
    sequence.printFinalAlignemnt()

iterations = [y[1] for y in pso_errors[0]]

pso_errors_no_iteration = [[y[0] for y in x] for x in pso_errors]
pso_errors_no_iteration2 = [[y[0] for y in x] for x in pso_errors2]
pso_errors_mean = np.mean(pso_errors_no_iteration, axis=0)
pso_errors_mean2 = np.mean(pso_errors_no_iteration2, axis=0)

pso_error_mean = np.mean(pso_error)
pso_error_mean2 = np.mean(pso_error2)
pso_error_std = np.std(pso_error)
pso_error_std2 = np.std(pso_error2)

plt.close()

fig = plt.figure()
plt.grid(1)
plt.xlim([0, ITERATIONS])
plt.ion()
plt.xlabel('Iterations')
plt.ylabel('Error')

plots = []
descriptions = []
plots.append(plt.plot(iterations, pso_errors_mean, 'b--', linewidth=1, markersize=3)[0])
plots.append(plt.plot(iterations, pso_errors_mean2, 'r--', linewidth=1, markersize=3)[0])
descriptions.append("Swarm 1 - Alignment fitness")
descriptions.append("Swarm 2 - Leading space fitness")

plt.legend(plots, descriptions)
fig.savefig('Result.png')
plt.show(5)

plt.close()

fig, ax = plt.subplots()

x = [x[0] for x in pso_error]
print(np.mean(x))
print('(' + str(np.std(x)) + ')')
y = [y[1] for y in pso_error]
print(np.mean(y))
print('(' + str(np.std(y)) + ')')

ax.scatter(x, y, color='blue', label="Swarm 1 - Alignment fitness")

x = [x[0] for x in pso_error2]
print(np.mean(x))
print('(' + str(np.std(x)) + ')')
y = [y[1] for y in pso_error2]
print(np.mean(y))
print('(' + str(np.std(y)) + ')')

ax.scatter(x, y, color='red', label="Swarm 2 - Leading space fitness")

ax.legend()
ax.grid()
fig.savefig('Result2.png')
plt.show(5)




