import numpy as np
import matplotlib.pyplot as plt
import polygon_tools as poly
import robot_tools
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors
import argparse
import yaml
from matplotlib.ticker import MultipleLocator
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean


""" 

Simple RRT implementation for visualising RRTs

Requires: numpy, matplotlib, argparse, pyyaml

Author: Nicholas Lawrance (nicholas.lawrance@mavt.ethz.ch)

"""

class RRTree:

    _start = None
    _goal = None
    _kdtree = None
    vertices = []
    edges = []

    def __init__(self, world, start=None, goal=None, delta=1.0, collision_step=0.01):
        self.world = robot_tools.PlanningProblem(world)
        self._delta = delta
        self.set_start(start)
        self.set_goal(goal)
        self._collision_step_dist = collision_step*euclidean(self.world.workspace.limits[:,0], self.world.workspace.limits[:,1])

    def set_start(self, start):
        assert len(start) == len(self.world.workspace.world_dims)
        self._start = np.array(start)

    def set_goal(self, goal):
        # Must be a poly.Polygon type
        self._goal = goal

    def search(self, imax = 1e4):
        self.vertices = [self._start]
        self.edges = []

        for i in range(0, imax):
            x_rand = self._sample_freespace()
            i_near = self._get_nearest(x_rand)
            x_near = self.vertices[i_near]
            x_new = self._steer(x_near, x_rand)
            if self._collision_free(x_near, x_new):
                self.vertices.append(x_new)
                self.edges.append([i_near, len(self.vertices)-1])
                self._kdtree = KDTree(self.vertices)


    def _sample_freespace(self, max_iter = 1e4):
        # Assume uniform sampling

        invalid_sample = True
        i = 0

        while invalid_sample and i < max_iter:
            x_rand = np.random.uniform(self.workspace.limits[:,0], self.workspace.limits[:,1])
            invalid_sample = self.workspace.in_collision(x_rand)
            i += 1

        if i == max_iter:
            raise Exception('Max iterations reached - no valid sample found')

        return x_rand


    def _get_nearest(self, point):
        # Lazy KDTRee from scipy
        dist, ind = self._kdtree.query(point, k=1)
        return ind

    def _steer(self, x_near, x_rand):
        dd = euclidean(x_near, x_rand)
        if dd < self._delta:
            return x_rand

        return (x_rand-x_near)/dd*self._delta

    def _collision_free(self, x_near, x_new):
        # Lazy collision walker








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Basic RRT example for plotting')
    parser.add_argument('-w', '--world', default='config/rrt_sample_world.yaml', help='World definition (obstacles)')
    args = parser.parse_args()

    arm_problem = robot_tools.PlanningProblem(args.world)



