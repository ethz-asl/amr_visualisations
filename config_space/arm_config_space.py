import numpy as np
import matplotlib.pyplot as plt
import polygon_tools as poly
import robot_tools
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import copy
import argparse

""" Plot the config space from Introduction to Autonomous Mobile Robots Fig 6.1 """

plt.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)

parser = argparse.ArgumentParser(description='Plot the config space from Fig 6.1 in Intro to AMR textbook')
parser.add_argument('-nx', type=int, default=101, help='Resolution (n points in each dimension')
args = parser.parse_args()

def make_rectangle_obstacle(xlim, ylim):
    return poly.Polygon([[xlim[0], ylim[0]], [xlim[1], ylim[0]], [xlim[1], ylim[1]], [xlim[0], ylim[1]]])


# Generate obstacles (random points then convex hull)
ob1 = make_rectangle_obstacle([2.3, 3.4], [7, 9.8])
ob2 = make_rectangle_obstacle([7.3, 8.5], [7.6, 10])
ob3 = make_rectangle_obstacle([1.3, 3], [2.8, 3.8])
ob4 = make_rectangle_obstacle([5, 7.1], [0.9, 3.7])

obstacles = [ob1, ob2, ob3, ob4]

arm = robot_tools.RobotArm2D(base_position=[5.0, 5.0], link_lengths=[2.1, 2.1])

theta1, theta2 = np.linspace(0, 2.0*np.pi, args.nx), np.linspace(0, 2.0*np.pi, args.nx)
v = np.zeros(((len(theta1), len(theta2))), dtype=int)

for i, t1 in enumerate(theta1):
    for j, t2 in enumerate(theta2):
        arm.set_link_angles([t1, t2])
        in_obs = 0
        fp = arm.get_current_polygon()
        for o_num, o in enumerate(obstacles):
            if fp.intersect(o):
                in_obs = o_num+1
                break
        v[i, j] = in_obs

f1, a1 = plt.subplots(1, 2)
h_obs = []
for o in obstacles:
    h_obs.append(PlotPolygon(o, color='lightgrey', zorder=1))
c_obs = PatchCollection(h_obs)
a1[0].add_collection(c_obs)

arm.set_link_angles([0.0*np.pi/180.0, 90.0*np.pi/180.0])

a1[0].plot(*arm.get_spine_points())

for ax in a1:
    ax.set_aspect('equal')

a1[0].set_xlabel(r'$x$')
a1[0].set_ylabel(r'$y$')
a1[0].set_xlim(0, 10)
a1[0].set_ylim(0, 10)

a1[1].set_xlabel(r'$\theta_1$')
a1[1].set_ylabel(r'$\theta_2$')
a1[1].set_xlim(0, theta2[-1])
a1[1].set_ylim(0, theta2[-1])

a1[1].matshow(v.transpose(), origin='lower', extent=[0, theta1[-1], 0, theta2[-1]], cmap='Greys')

plt.show()