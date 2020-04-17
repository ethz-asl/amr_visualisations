import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class RotationAnimator(object):
    h_surf = None
    h_goal = None

    def __init__(self, x, y, z, goal, start_azim=-60.0, delta_angle=10.0, start_elev=30.0):
        self.f = plt.figure()
        self.a = self.f.add_subplot(111, projection='3d')
        self.x = x
        self.y = y
        self.z = z
        self.goal = goal
        self.start_azim = start_azim
        self.start_elev = start_elev
        self.delta_angle = delta_angle
        self.init()

    def init(self):
        self.a.cla()
        self.h_surf = self.a.plot_surface(self.x, self.y, self.z.T, cmap=cm.coolwarm)
        # self.h_start = a3.plot([self.start[0]], [self.start[1]], [self.z[self.start[0], self.start[1]]], 'g^')
        # self.h_path = a3.plot(self.path[:,0], self.path[:,1])
        self.h_goal, = self.a.plot([self.goal[0]], [self.goal[1]], [self.z[int(self.goal[0]), int(self.goal[1])]], 'yo')
        self.a.azim = self.start_azim
        self.a.elev = self.start_elev
        return [self.h_surf, self.h_goal]

    def update(self, n):
        self.a.view_init(elev=self.start_elev, azim=self.start_azim+n*self.delta_angle)
        return [self.h_surf, self.h_goal]


class TrisurfRotationAnimator(object):
    h_surf = None

    def __init__(self, vertices, faces, ax_lims=None, start_azim=-60.0, delta_angle=10.0, start_elev=30.0,
                 x_label=r'$x$', y_label=r'$y$', z_label=r"$z$"):
        self.f = plt.figure()
        self.a = self.f.add_subplot(111, projection='3d')
        self.vertices = vertices
        self.faces = faces
        self.start_azim = start_azim
        self.start_elev = start_elev
        self.delta_angle = delta_angle
        self.ax_lims = ax_lims
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self.init()

    def init(self):
        self.a.cla()
        self.a.set_xlabel(self.x_label)
        self.a.set_ylabel(self.y_label)
        self.a.set_zlabel(self.z_label)

        self.h_surf = self.a.plot_trisurf(self.vertices[:, 0], self.vertices[:, 1], self.faces, self.vertices[:, 2],
                        cmap='Spectral', lw=1)
        if self.ax_lims is not None:
            self.a.set_xlim(self.ax_lims[0])
            self.a.set_ylim(self.ax_lims[1])
            self.a.set_zlim(self.ax_lims[2])

        self.a.azim = self.start_azim
        self.a.elev = self.start_elev
        return [self.h_surf]

    def update(self, n):
        self.a.view_init(elev=self.start_elev, azim=self.start_azim+n*self.delta_angle)
        return [self.h_surf]
