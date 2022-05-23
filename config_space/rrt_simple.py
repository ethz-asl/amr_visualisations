import time
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

import polygon_tools
import robot_tools
from matplotlib.collections import LineCollection
from scipy.spatial import KDTree, Voronoi, voronoi_plot_2d, distance


""" 

Simple RRT implementation for visualising RRTs with Voronoi regions

Requires: numpy, scipy, matplotlib, argparse, pyyaml

Author: Nicholas Lawrance (nicholas.lawrance@mavt.ethz.ch)

"""


class RRTree(object):
    _start = None
    _goal = None
    _kdtree = None
    _kd = False
    vertices = []
    edges = []

    def __init__(self, world, start=None, goal=None, delta=1.0,
                 collision_step=0.01, nearest='cdist', p_goal=0.0):
        self.world = robot_tools.PlanningProblem(world)
        logging.info('Loaded RRT search problem from {0}'.format(world))
        self._delta = delta
        self._p_goal = p_goal
        if nearest == 'kdtree':
            self._get_nearest = self._get_nearest_kd
            self._kd = True
            logging.info('Using kd tree nearest-neighbour search')
        else:
            self._get_nearest = self._get_nearest_cdist
            logging.info('Using scipy cdist nearest-neighbour search')

        if start is not None:
            self.set_start(start)
        if goal is not None:
            self.set_goal(goal)
        self._collision_step_dist = collision_step*distance.euclidean(self.world.workspace.limits[:,0], self.world.workspace.limits[:,1])

    def set_start(self, start):
        # Slightly hacky because we are assuming planning in workspace, not config space
        assert len(start) == len(self.world.workspace.limits)
        self._start = np.array(start)

    def set_goal(self, goal):
        # Must be a poly.Polygon type
        assert len(goal) == len(self.world.workspace.limits)
        self._goal = np.array(goal)

    def search(self, imax = 1000):
        self.vertices = [self._start]
        self.edges = [[]]
        if self._kd:
            self._kdtree = KDTree(self.vertices)

        for i in range(0, imax):
            goal_found = self._search_step()
            if ~goal_found:
                break

    def _search_step(self):
        goal_found = False
        x_rand = self._sample_freespace()
        i_near = self._get_nearest(x_rand)
        x_near = self.vertices[i_near]
        x_new = self._steer(x_near, x_rand)
        if self._valid_pose(x_new) and self._collision_free(x_near, x_new):
            if (x_new == self._goal).all():
                goal_found = True
            self.vertices.append(x_new)
            self.edges.append([])
            self.edges[i_near].append(len(self.vertices) - 1)
            if self._kd:
                self._kdtree = KDTree(self.vertices)
        return goal_found

    # def animated_search(self, imax=100, show_voronoi = True):

    def _valid_pose(self, x):
        self.world.robot.set_position(x)
        return not self.world.workspace.in_collision_poly(self.world.robot.get_current_polygon())

    def _sample_freespace(self, max_iter=1000):
        # Assume uniform sampling
        valid = False
        i = 0
        while (not valid) and i < max_iter:
            if 0 < self._p_goal < np.random.uniform():
                x_rand = self._goal
            else:
                x_rand = np.random.uniform(self.world.workspace.limits[:, 0], self.world.workspace.limits[:, 1])
            valid = self._valid_pose(x_rand)
            i += 1

        if i == max_iter:
            raise Exception('Max iterations reached - no valid sample found')

        return x_rand

    def _get_nearest_kd(self, point):
        # Lazy KDTRee from scipy - deprecated (cdist is much faster)
        dist, ind = self._kdtree.query(point, k=1)
        return ind

    def _get_nearest_cdist(self, point):
        d_full = distance.cdist(self.vertices, [point]).flatten()
        i_best = np.argmin(d_full)
        return i_best

    def _steer(self, x_near, x_rand):
        dd = distance.euclidean(x_near, x_rand)
        if dd < self._delta:
            return x_rand
        return x_near + (x_rand-x_near)/dd*self._delta

    def _collision_free(self, x_near, x_new, dist=None):
        # Lazy collision walker
        valid = True
        d_full = distance.euclidean(x_new, x_near)
        v_hat = (x_new - x_near)/d_full
        for i in range(1, int(d_full/self._collision_step_dist)+1):
            c_pos = x_near + i*self._collision_step_dist*v_hat
            valid = self._valid_pose(c_pos)
            if not valid:
                break
        return valid

    def plot_tree(self, ax):
        ax.plot([self._start[0]], [self._start[1]], 'go')
        h_verts = plt.plot([v[0] for v in self.vertices], [v[1] for v in self.vertices], 'k.')

        lines = []
        for i_start in range(len(self.edges)):
            for i_stop in self.edges[i_start]:
                lines.append([self.vertices[i_start], self.vertices[i_stop]])
        lc = LineCollection(lines, linewidths=2, colors='firebrick')
        ax.add_collection(lc)
        return h_verts, lc

    def plot_voronoi(self, ax, **kwargs):
        vor = Voronoi(self.vertices)
        voronoi_plot_2d(vor, ax=ax, **kwargs)


if __name__ == "__main__":
    plt.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Roman']})
    plt.rc('text', usetex=True)

    parser = argparse.ArgumentParser(description='Basic RRT example for plotting')
    parser.add_argument('-w', '--world', default='config/rrt_sample_world.yaml', help='World definition (obstacles)')
    parser.add_argument('-i', '--iterations', type=int, default=100, help='Max RRT iterations')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--save-fig', action='store_true', help='Save figure location')
    parser.add_argument('-kd', action='store_true', help='Use kd_tree (it is outright slower though!)')
    parser.add_argument('--pgoal', type=float, default=0.0, help='Probability of attempting to sample goal directly')
    parser.add_argument('--loglevel', default='INFO', type=str,
                        help='World definition (obstacles)')
    args = parser.parse_args()
    np.random.seed(args.seed)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    if args.kd:
        nearest = 'kdtree'
    else:
        nearest = 'cdist'

    rrt = RRTree(args.world, nearest=nearest, p_goal=args.pgoal)

    start_pos = [5.0, 5.0]
    goal_pos = [9.5, 9.5]
    rrt.set_start(start_pos)
    rrt.set_goal(goal_pos)

    t0 = time.time()
    logging.info('Running RRT search with max iterations n={0}'.format(args.iterations))
    rrt.search(imax=args.iterations)
    logging.info('RRT search complete in {0}s ({1})'.format(time.time()-t0, nearest))
    f = plt.figure(figsize=(6, 6))
    ax = f.add_subplot(1, 1, 1)
    if len(rrt.vertices) >= 4:
        rrt.plot_voronoi(ax, show_points=False, show_vertices=False, line_colors='grey')
    rrt.world.workspace.plot(ax)
    rrt.plot_tree(ax)

    if args.save_fig:
        f.savefig('fig/rrt/{0:04}.png'.format(args.iterations), bbox_inches='tight')
    else:
        plt.show()


