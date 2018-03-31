import numpy as np

from mlpy.particleSwarmOptimization.pso import PSO
from mlpy.particleSwarmOptimization.structure.particle import Particle

class VectorEvaluatedPSO(PSO):

    def __init__(self):
        super(VectorEvaluatedPSO, self).__init__()

        self.group_best_error2 = float('inf')
        self.group_best_position2 = None

        self.best_error2 = float('inf')
        self.best_position2 = None

        self.swarm2 = []

        self.error2 = None

    def getGlobalBest(self):
        self.best_error = float('inf')
        for particle in self.swarm:
            if particle.error < self.group_best_error:
                self.group_best_position = np.array(particle.position)
                self.group_best_error = particle.error
            # Get current best as well
            if particle.error < self.best_error:
                self.best_position = np.array(particle.position)
                self.best_error = particle.error

        self.best_error2 = float('inf')
        for particle in self.swarm2:
            if particle.error < self.group_best_error2:
                self.group_best_position2 = np.array(particle.position)
                self.group_best_error2 = particle.error
            # Get current best as well
            if particle.error < self.best_error2:
                self.best_position2 = np.array(particle.position)
                self.best_error2 = particle.error

        return self.group_best_position

    def createParticles(self):
        for i in range(self.num_particles):
            self.swarm.append(Particle(self.bounds, self.weight, self.cognitiveConstant, self.socialConstant))
            position = (self.initialPosition.maxBound - self.initialPosition.minBound) * np.random.random(
                self.num_dimensions) + self.initialPosition.minBound
            velocity = np.zeros(self.num_dimensions)
            self.swarm[i].initPos(position, velocity)

        for i in range(self.num_particles):
            self.swarm2.append(Particle(self.bounds, self.weight, self.cognitiveConstant, self.socialConstant))
            position = (self.initialPosition.maxBound - self.initialPosition.minBound) * np.random.random(
                self.num_dimensions) + self.initialPosition.minBound
            velocity = np.zeros(self.num_dimensions)
            self.swarm2[i].initPos(position, velocity)

    def loopOverParticles(self):
        for j in range(self.num_particles):
            self.swarm[j].error = self.error(self.swarm[j].position)

            self.swarm[j].getPersonalBest()

        for j in range(self.num_particles):
            self.swarm2[j].error = self.error2(self.swarm[j].position)

            self.swarm2[j].getPersonalBest()

        self.getGlobalBest()

        for j in range(self.num_particles):
            self.swarm[j].update_velocity(self.group_best_position2)
            self.swarm[j].update_position()

            self.swarm2[j].update_velocity(self.group_best_position)
            self.swarm2[j].update_position()

            self.swarm[j].position = np.clip(self.swarm[j].position, self.bounds.minBound, self.bounds.maxBound)
            self.swarm2[j].position = np.clip(self.swarm[j].position, self.bounds.minBound, self.bounds.maxBound)


    def train(self, iterations, sampleSize):
        self.createParticles()

        trainingErrors = []

        for x in range(iterations):

            self.loopOverParticles()

            if (x % sampleSize == 0):
                trainingErrors.append([self.group_best_error, x])

        trainingErrors.append([self.group_best_error, iterations])

        return trainingErrors, self.group_best_error
