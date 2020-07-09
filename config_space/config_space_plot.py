from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import polygon_tools as poly
import robot_tools
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import copy
import argparse
from plot_tools.surf_rotation_animation import TrisurfRotationAnimator

""" 

Plot an example of config space for Autonomous Mobile Robots lecture notes 

Requires: numpy, matplotlib, argparse, scikit-image (>=0.13, for marching cubes)

Author: Nicholas Lawrance (nicholas.lawrance@mavt.ethz.ch)

"""

plt.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)

parser = argparse.ArgumentParser(description='Basic visualisation of configuration space for mobile robot')
parser.add_argument('-nx', type=int, default=61, help='Resolution (n points in each dimension')
parser.add_argument('-rf', '--robot-footprint', default='config/bar_robot.csv', help='Robot footprint csv file')
parser.add_argument('-no', '--n-obstacles', type=int, default=5, help='Number of obstacles')
parser.add_argument('-ns', '--n-samples', type=int, default=5, help='Number of sample locations for testing')
parser.add_argument('-ss', '--std-samples', type=float, default=0.1, help='Sample standard deviation')
parser.add_argument('--seed', type=int, default=5, help='Numpy random seed')
parser.add_argument('--animation', action='store_true', help='Generate animation')
args = parser.parse_args()

nx = args.nx
num_obstacles = args.n_obstacles
n_obs_samples = args.n_samples
obs_std = args.std_samples
np.random.seed(args.seed)

# Generate obstacles (random points then convex hull)
obs_centres = [poly.Point(*np.random.uniform(size=2)) for i in range(num_obstacles)]
obstacles = []
for pc in obs_centres:
    px, py = np.random.normal(pc, obs_std, size=(n_obs_samples, 2)).T
    px, py = np.clip(px, 0.0, 1.0), np.clip(py, 0.0, 1.0)
    p = poly.PointList([poly.Point(x, y) for x, y in zip(px, py)])
    p = poly.convex_hull(p)
    obstacles.append(p)

# Get some random points and see if they're in the obstacles:
in_obs, out_obs = poly.PointList([]), poly.PointList([])
for i in range(200):
    p = poly.Point(*np.random.uniform(size=2))
    collision = False
    for o in obstacles:
        if o.point_inside(p):
            collision = True
            break
    if collision:
        in_obs.append(p)
    else:
        out_obs.append(p)

f1, a1 = plt.subplots()
h_obs = []
for o in obstacles:
    h_obs.append(PlotPolygon(o, color='lightgrey', zorder=1))
c_obs = PatchCollection(h_obs)
a1.add_collection(c_obs)
a1.scatter(*zip(*in_obs), color='r', marker='x')
a1.scatter(*zip(*out_obs), color='g', marker='.')
print("Intersect: {0}".format(obstacles[0].intersect(obstacles[1])))

# Load the robot shape
robo = robot_tools.Robot2D(footprint_file=args.robot_footprint)

# Now try robot poses:
a1.add_artist(PlotPolygon(robo.get_current_polygon(), facecolor='r'))

robo.set_position((0.25, 0.38))
robo.get_current_polygon().intersect(obstacles[-1])

x, y, h = np.linspace(0, 1, nx), np.linspace(0, 1, nx), np.linspace(0, np.pi, nx)
v = np.zeros((len(x), len(y), len(h)))
for i,xi in enumerate(x):
    for j, yj in enumerate(y):
        robo.set_position((xi, yj))
        for k, hk in enumerate(h):
            in_obs = 0.0
            robo.set_heading(hk)
            fp = robo.get_current_polygon()
            for o in obstacles:
                if fp.intersect(o):
                    in_obs = 1.0
                    break
            v[i, j, k] = in_obs

verts, faces, normals, values = measure.marching_cubes_lewiner(v, spacing=(x[1]-x[0], y[1]-y[0], (h[1]-h[0])*180/np.pi))
ax_lims = [[0, x[-1]], [0, y[-1]], [0, h[-1]*180/np.pi]]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1)
ax.set_xlim(ax_lims[0])
ax.set_ylim(ax_lims[1])
ax.set_zlim(ax_lims[2])
ax.set_xlabel(r'$x_c$')
ax.set_ylabel(r'$y_c$')
ax.set_zlabel(r"$\theta (^{\circ})$")

robo.set_position([0.1, 0.1])
f2, a2 = plt.subplots(2, 2)
for i, ax in enumerate(a2.flat):
    dex = int(i*0.25*(len(h)-1))
    ax.matshow(v[:, :, dex].transpose(), origin='lower', extent=[0, 1, 0, 1], cmap='Greys')
    ax.add_collection(PatchCollection(copy.copy(h_obs)))
    robo.set_heading(h[dex])
    ax.add_artist(PlotPolygon(robo.get_current_polygon(), facecolor='r'))
    ax.plot(*robo.position, color='g', marker='x')
    ax.set_title(r"$\theta = {0:0.1f}$".format(h[dex]*180/np.pi))
    ax.tick_params(top=0, left=0)

if args.animation:
    rotator = TrisurfRotationAnimator(verts, faces, ax_lims=ax_lims, delta_angle=5.0,
                                      x_label=r'$x_c$', y_label=r'$y_c$', z_label=r"$\theta (^{\circ})$")
    ani = animation.FuncAnimation(rotator.f, rotator.update, 72, init_func=rotator.init, interval=10, blit=False)
    # ani.save('fig/config_space_rotation.gif', writer='imagemagick', fps=15)
    ani.save('fig/config_space_rotation.mp4', writer='ffmpeg', fps=int(15),
                       extra_args=["-crf", "18", "-profile:v", "main", "-tune", "animation", "-pix_fmt", "yuv420p"])

plt.show()
