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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Basic RRT example for plotting')
    parser.add_argument('-w', '--world', default='config/rrt_sample.yaml', help='World definition (obstacles)')
    args = parser.parse_args()

    arm_problem = PlanningProblem(args.world)