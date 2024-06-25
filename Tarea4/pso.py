import sys
import os
import copy

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from matplotlib import animation


def rastrigin(point):
    return sum([pow(pi, 2) - 10 * np.cos(2 * 3.14 * pi) for pi in point]) + 20


class Particle:
    def __init__(self, pos, velocity):
        self.pos = pos
        self.velocity = velocity
        self.fitness = rastrigin(self.pos)

        self.best_pos = np.copy(self.pos)
        self.best_fitness = self.fitness
        
    def __repr__(self):
        return f'Particle(\'{self.pos}\', {self.velocity}, {self.best_fitness})'


def pso(max_iter=10, maxp=10, minp=-10):
    # hyper-parameters
    w1 = 0.729
    w2 = 1.49445
    w3 = 1.49445

    swarm = [Particle([0, 6], [0, 9 / 2]), Particle([4, 0], [9 / 2, 0]),
             Particle([0, -6], [0, 9 / 2]), Particle([-4, 0], [9 / 2, 0])]
    swarm_size = len(swarm)

    best_swarm_fitness = sys.float_info.max
    for i in range(swarm_size):
        if swarm[i].fitness < best_swarm_fitness:
            best_swarm_fitness = swarm[i].fitness
            best_swarm_pos = swarm[i].pos

    history = {'swarm': [],
               'best_swarm_fitness': [],
               'best_swarm_pos': [[np.inf, np.inf] for _ in range(max_iter)],
               'obj_func': 'rastrigin'}

    for itr in range(max_iter):
        history['swarm'].append(copy.deepcopy(swarm))
        history['best_swarm_fitness'].append(best_swarm_fitness)
        history['best_swarm_pos'][itr][0] = best_swarm_pos[0]
        history['best_swarm_pos'][itr][1] = best_swarm_pos[1]

        for i in range(swarm_size):
            # new velocity
            for d in range(2):                
                swarm[i].velocity[d] = (
                    (w1 * swarm[i].velocity[d]) +
                    (w2 * np.random.randint(swarm_size) * (swarm[i].best_pos[d] - swarm[i].pos[d])) +
                    (w3 * np.random.randint(swarm_size) *
                     (best_swarm_pos[d] - swarm[i].pos[d]))
                )
                
                if swarm[i].velocity[d] < minp:
                    swarm[i].velocity[d] = minp
                elif swarm[i].velocity[d] > maxp:
                    swarm[i].velocity[d] = maxp

            # new pos
            swarm[i].pos[0] += swarm[i].velocity[0]
            swarm[i].pos[1] += swarm[i].velocity[1]

            # new fitness
            swarm[i].fitness = rastrigin(swarm[i].pos)

            if swarm[i].fitness < swarm[i].best_fitness:
                swarm[i].best_pos = swarm[i].pos
                swarm[i].best_fitness = swarm[i].fitness
            if swarm[i].fitness < best_swarm_fitness:
                best_swarm_fitness = swarm[i].fitness
                best_swarm_pos = swarm[i].pos
    return history


def visualization(history=None, bounds=None, minima=None):

    # define meshgrid according to given boundaries
    x = np.linspace(bounds[0][0], bounds[0][1], 50)
    y = np.linspace(bounds[1][0], bounds[1][1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.array([rastrigin([x, y]) for x, y in zip(X, Y)])

    # initialize figure
    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(121, facecolor='w')
    ax2 = fig.add_subplot(122, facecolor='w')

    # animation callback function
    def animate(frame, history):
        # print('current frame:',frame)
        ax1.cla()
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_title('{}|iter={}|best pos=({:.5f},{:.5f})'.format("rastrigin", frame+1,
                      history['best_swarm_pos'][frame][0], history['best_swarm_pos'][frame][1]))
        ax1.set_xlim(bounds[0][0], bounds[0][1])
        ax1.set_ylim(bounds[1][0], bounds[1][1])
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fitness')
        ax2.set_title('Minima Value Plot|Population={}|MinVal={:}'.format(
            len(history['swarm'][0]), history['best_swarm_fitness'][frame]))
        ax2.set_xlim(2, len(history['best_swarm_fitness']))
        ax2.set_ylim(10, 10)
        ax2.set_yscale('log')

        # data to be plot
        data = history['swarm'][frame]
        best_swarm_pos = np.array(history['best_swarm_fitness'])

        # contour and global minimum
        contour = ax1.contour(X, Y, Z, levels=50, cmap="magma")
        ax1.plot(minima[0], minima[1], marker='o', color='black')
        # plot swarm
        data_x = [data[n].pos[0] for n in range(len(data))]
        data_y = [data[n].pos[1] for n in range(len(data))]
        ax1.scatter(data_x, data_y, marker='x', color='black')
        if frame > 1:
            for i in range(len(data)):
                ax1.plot([history['swarm'][frame-n][i].pos[0] for n in range(2, -1, -1)],
                         [history['swarm'][frame-n][i].pos[1] for n in range(2, -1, -1)])
        elif frame == 1:
            for i in range(len(data)):
                ax1.plot([history['swarm'][frame-n][i].pos[0] for n in range(1, -1, -1)],
                         [history['swarm'][frame-n][i].pos[1] for n in range(1, -1, -1)])

        # plot current global best
        x_range = np.arange(1, frame+2)
        ax2.plot(x_range, best_swarm_pos[0:frame+1])

    # title of figure
    fig.suptitle('Optimizing of {} function by PSO'.format("rastrigin".split()[0]), fontsize=20)

    ani = animation.FuncAnimation(fig, animate, fargs=(history,),
                                  frames=len(history['swarm']), interval=250, repeat=False, blit=False)

    os.makedirs('gif/', exist_ok=True)
    ani.save('gif/PSO_{}_population_{}.gif'.format("rastrigin".split()
             [0], len(history['swarm'][0])), writer="imagemagick")
    print('A gif video is saved at gif/')


history = pso()
print('best_swarm_fitness:', history['best_swarm_fitness']
      [-1], ', best_swarm_pos:', history['best_swarm_pos'][-1])
visualization(history=history, bounds=[[-20, 20], [-20, 20]], minima=[0, 0])
