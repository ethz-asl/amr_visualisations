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


""" 

Simple RRT implementation for visualising RRTs

Requires: numpy, matplotlib, argparse, pyyaml

Author: Nicholas Lawrance (nicholas.lawrance@mavt.ethz.ch)

"""

class RRTree:

    _start = None
    _goal = None

    def __init__(self, world, start=None, goal=None, collision_checker=):
        self.world = world
        self.set_start(start)
        self.set_goal(goal)

    def set_start(self, start):
        assert len(start) == len(self.world.world_dims)
        self._start = start

    def set_goal(self, goal):
        self._goal = goal

    def search(self, imax = 1e4):
        G = [[]]

        for i in range(0, imax):
            x_rand = self.sample_freespace()


    def sample_freespace(self):
        # Assume uniform sampling

        valid_sample = False

        while not valid_sample:
            x_rand = np.random.uniform(self.world.world_dims)
        pass








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Basic RRT example for plotting')
    parser.add_argument('-w', '--world', default='config/rrt_sample_world.yaml', help='World definition (obstacles)')
    args = parser.parse_args()

    arm_problem = robot_tools.PlanningProblem(args.world)



