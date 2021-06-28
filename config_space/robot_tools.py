import numpy as np
import polygon_tools as poly
import csv
import yaml
import sys
import matplotlib as plt
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm


def robot_builder(robot):
    # Note that the robot type must be implemented in this module, so the example robot:
    #  {type: RobotArm2D, parameters: {base_position: [5.0, 5.0], link_lengths: [2.1, 2.1]}
    # would call as constructor: robot_tools.RobotArm2D(base_position=[5.0, 5.0], link_lengths=[2.1, 2.1])
    return getattr(sys.modules[__name__], robot['type'])(**robot['parameters'])


def workspace_builder(workspace):
    # Note that the workspace type must be implemented in this module, so the example workspace:
    #  {type: Workspace2D, parameters: {limits=[[0,1.0],[0,1.0]], obstacles=[]}
    # would call as constructor: robot_tools.Workspace2D(limits=[[0,1.0],[0,1.0]], obstacles=[])
    return getattr(sys.modules[__name__], workspace['type'])(**workspace['parameters'])


class Workspace2D(object):
    def __init__(self, limits=[[0,1.0],[0,1.0]], obstacles=[]):
        self.limits = np.array(limits)
        assert self.limits.shape == (2,2), 'Currently only implemented for 2D workspaces'
        self.obstacles = []
        for ob in obstacles:
            # Add each obstacle (must be a Polygon or derived class like Rectangle from poly_tools)
            self.obstacles.append(getattr(poly, ob['type'])(**ob['parameters']))

    def in_collision_point(self, point):
        p = poly.Point(*point)
        collision = False
        for o in self.obstacles:
            if o.point_inside(p):
                collision = True
                break
        return collision

    def in_collision_poly(self, polygon):
        collision = False
        for o in self.obstacles:
            if polygon.intersect(o):
                collision = True
                break
        return collision

    def plot(self, hax=None, cmap=cm.viridis):
        if hax is None:
            f, hax = plt.subplots(1)
        h_obs = []
        for o in self.obstacles:
            h_obs.append(PlotPolygon(o, zorder=1))
        c_obs = PatchCollection(h_obs, cmap=cmap)
        # This sets colors for some reason (command in Polygon does not)
        c_obs.set_array(np.linspace(0, 1.0, len(self.obstacles) + 1)[1:])
        hax.add_collection(c_obs)

        hax.set_aspect('equal')

        hax.set_xlabel(r'$x$')
        hax.set_ylabel(r'$y$')
        hax.set_xlim(self.limits[0])
        hax.set_ylim(self.limits[1])


class Robot2D(object):
    _c_poly = None

    def __init__(self, pos=[0.0, 0.0], heading=0.0, footprint=[(0.0, 0.0)], footprint_file=None):
        self.R = np.eye(2)

        if footprint_file is not None:
            with open(footprint_file, mode='r') as fh:
                csv_reader = csv.reader(fh)
                footprint = []
                for row in csv_reader:
                    assert len(row) == 2, 'Row {0} does not have 2 elements'.format(len(row)+1)
                    footprint.append([float(row[0]), float(row[1])])
            print('Loaded robot footprint file {0} with {1} points'.format(footprint_file, len(footprint)))

        self.position = poly.Point(pos[0], pos[1])
        self.footprint = poly.PointList(footprint)
        self.heading = heading
        self._set_heading_transformation()
        self._update_poly()


    def _set_heading_transformation(self):
        ct, st = np.cos(self.heading), np.sin(self.heading)
        self.R = np.array([[ct, -st], [st, ct]])

    def set_heading(self, heading):
        self.heading = heading
        self._set_heading_transformation()
        self._update_poly()

    def set_position(self, pos):
        self.position = pos
        self._update_poly()

    def set_footprint(self, footprint):
        self.footprint = footprint
        self._update_poly()

    def _update_poly(self):
        self._c_poly = poly.Polygon([poly.Point(*(np.matmul(self.R, p)+self.position)) for p in self.footprint])

    def get_current_polygon(self):
        return self._c_poly


class RobotArm2D(object):
    _spine_pts = None

    def __init__(self, base_position=[0.0, 0.0], link_lengths=[1.0, 1.0], link_angles=[0.0, 0.0]):
        # Assume arm angles are relative (can be summed)

        self._base_position = poly.Point(base_position[0], base_position[1])

        assert len(link_lengths) == len(link_angles)
        self._link_lengths = np.array(link_lengths)
        self._link_angles = np.array(link_angles)

        self._R = [np.eye(2) for i in self._link_angles]
        self._set_rotation_transforms()

    def set_link_angles(self, link_angles):
        self._link_angles = np.array(link_angles)
        self._set_rotation_transforms()

    def _set_rotation_transforms(self):
        sum_angles = self._link_angles.cumsum()
        for i, theta in enumerate(sum_angles):
            ct, st = np.cos(theta), np.sin(theta)
            self._R[i] = np.array([[ct, -st], [st, ct]])
        self._set_spine_points()

    def get_current_polygon(self):
        # Run backwards through the points to make a polygon
        return poly.Polygon(self._spine_pts + self._spine_pts[-2:0:-1])

    def _set_spine_points(self):
        self._spine_pts = [self._base_position]
        for R, ll in zip(self._R, self._link_lengths):
            self._spine_pts.append(poly.Point(*(np.matmul(R, [ll, 0])+self._spine_pts[-1])))

    def get_spine_points(self):
        return [p.x for p in self._spine_pts], [p.y for p in self._spine_pts]

    def get_end_effector_position(self):
        return self._spine_pts[-1]

    def end_effector_path(self, config_path):
        c_pose = self._link_angles.copy()
        ee_path = []
        for pose in config_path:
            self.set_link_angles(pose)
            ee_path.append(self.get_end_effector_position())
        self.set_link_angles(c_pose)
        return np.array(ee_path)


class PlanningProblem(object):

    def __init__(self, world_file):

        # Load world
        with open(world_file, 'r') as fh:
            world = yaml.safe_load(fh)

        self.workspace = workspace_builder(world['workspace'])
        # Note that the robot type must be implemented in the robot_tools module, so the example robot:
        #  {type: RobotArm2D, parameters: {base_position: [5.0, 5.0], link_lengths: [2.1, 2.1]}
        # would call as constructor: robot_tools.RobotArm2D(base_position=[5.0, 5.0], link_lengths=[2.1, 2.1])
        self.robot = robot_builder(world['robot'])


    def construct_config_space(self, nx=101):
        # TODO: This should be more general (number of dimensions, wraparound etc. in the robot class)
        theta1, theta2 = np.linspace(0, 2.0 * np.pi, nx), np.linspace(0, 2.0 * np.pi, nx)
        v = np.zeros((len(theta1), len(theta2)), dtype=int)

        for i, t1 in enumerate(theta1):
            for j, t2 in enumerate(theta2):
                self.robot.set_link_angles([t1, t2])
                in_obs = 0
                fp = self.robot.get_current_polygon()
                for o_num, o in enumerate(self.workspace.obstacles):
                    if fp.intersect(o):
                        in_obs = o_num + 1
                        break
                v[i, j] = in_obs

        return [theta1, theta2], v