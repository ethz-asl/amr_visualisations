import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.patches import Circle
import matplotlib.animation as animation
# import rotanimate

# Annoying Windows stuff if running on Windows (requires pointer to ImageMagick)
# import os, sys
# imgk_path = os.path.join('C:/', 'Users', 'lawrancn', 'Downloads', 'ImageMagick', 'convert.exe')
# plt.rcParams['animation.convert_path'] = imgk_path
# if imgk_path not in sys.path: sys.path.append(imgk_path)

class World(object):
    def __init__(self, width, height, max_cost=1.0):
        self.width = width
        self.height = height
        self.cost_map = np.zeros((height, width), dtype='float')
        self.obs_map = np.zeros((height, width), dtype='bool')
        self.obs = []
        self.global_cost = []
        self.max_cost = max_cost

    def reset(self):
        self.obs = []
        self.global_cost = []
        self.cost_map *= 0
        self.obs_map *= 0

    def add_obstacles(self, centres, radii, Q_star, eta):
        for c, r in zip(centres, radii):
            new_obstacle = RoundObstacle(c, r, Q_star, eta)
            self.obs.append(new_obstacle)

        # Probably a bit silly, but we can cheat because we have circular objects:
        for i in range(self.width):
            for j in range(self.height):
                # Get distances to all obstacle boundaries
                d_obs = np.linalg.norm(centres - np.array([i, j]), axis=1) - radii
                nearest_obs = np.argmin(d_obs)
                d_obs = d_obs[nearest_obs]
                if d_obs <= 0:
                    self.obs_map[i, j] = 1
                    self.cost_map[i, j] = self.max_cost
                else:
                    self.cost_map[i, j] += min(self.max_cost, self.obs[nearest_obs].cost(d_obs))

    def add_global_cost(self, global_cost):
        self.global_cost = global_cost
        for i in range(self.width):
            for j in range(self.height):
                self.cost_map[i,j] += self.global_cost.get_cost([i, j])
                self.cost_map[i,j] = min(self.cost_map[i,j], self.max_cost)


class RoundObstacle(object):
    def __init__(self, pos, r, Q_star, eta):
        self.pos = pos
        self.r = r
        self.Q_star = Q_star
        self.half_eta = 0.5*eta
        self.inv_Qstar = 1.0/Q_star

    def cost(self, d):
        if d <= 0 or d > self.Q_star:
            u = 0.0
        else:
            u = self.half_eta*(1.0/d - self.inv_Qstar)**2
        return u

    def get_cost(self, p):
        d = np.linalg.norm(self.pos - p ) - self.r
        return self.cost(d)


class CombinedGlobalPotential(object):
    def __init__(self, goal, zeta, d_star):
        self.goal = goal
        self.zeta = zeta
        self.d_star = d_star
        self.offset = 0.5*self.zeta*self.d_star**2

    def get_cost(self, p):
        d = np.linalg.norm(p-self.goal)
        if d <= self.d_star:
            u = 0.5*self.zeta*d**2
        else:
            u = self.d_star*self.zeta*d -self.offset
        return u


def gradient_path(dx, dy, start, goal, step=0.1, goal_range = 0.5):
    path = [start.copy()]
    cp = start.copy()
    while np.linalg.norm(goal-cp) > goal_range:
        d_int = np.rint(cp).astype(int)
        ddx, ddy = dx[d_int[0], d_int[1]], dy[d_int[0], d_int[1]]
        ddd = np.sqrt(ddx**2 + ddy**2)
        cp[0] = cp[0] - step*ddx/ddd
        cp[1] = cp[1] - step*ddy/ddd
        path.append(cp.copy())
    return np.array(path)


class RotationAnimator(object):
    def __init__(self, x, y, z, goal, start_azim=-60.0, delta_angle=10.0, start_elev=30.0):
        self.f = plt.figure()
        self.a = self.f.add_subplot(111, projection='3d')
        self.h_surf = a3.plot_surface(x, y, z.T, cmap=cm.coolwarm)
        self.x = x
        self.y = y
        self.z = z
        self.goal = goal
        self.h_goal = a3.plot([goal[0]], [goal[1]], [z[int(goal[0]), int(goal[1])]], 'yo')
        # self.h_start = a3.plot([start[0]], [start[1]], [z[start[0], start[1]]], 'g^')
        # self.h_path = a3.plot(path[:,0], path[:,1])
        self.a.azim = start_azim
        self.a.elev = start_elev
        self.start_azim = start_azim
        self.start_elev = start_elev
        self.delta_angle = delta_angle

    def update_anim(self, n):
        self.a.cla()
        h_surf = self.a.plot_surface(self.x, self.y, self.z.T, cmap=cm.coolwarm)
        # self.h_goal = a3.plot([self.goal[0]], [self.goal[1]], [self.z[int(self.goal[0]), int(self.goal[1])]], 'yo')
        self.a.azim = self.start_azim+n*self.delta_angle
        self.a.elev = self.start_elev
        return h_surf,


w = 100
h = 100
n_obs = 5
goal = np.array([5.0, 5.0])
start = np.array([70.0, 90.0])

my_world = World(w, h)

# centres = np.zeros((n_obs, 2))
# centres[:, 0] = np.random.uniform(0, w, n_obs)
# centres[:, 1] = np.random.uniform(0, h, n_obs)
# ranges = np.random.uniform(5.0, 15.0, n_obs)
centres = np.array([[20, 30], [80, 10], [10, 78.8], [90.5, 75]])
ranges = np.array([5.0, 15.0, 12.0, 11.0])

my_world.add_obstacles(centres, ranges, Q_star=15.0, eta=15.0)

fo, ho = plt.subplots()
ho.matshow(my_world.cost_map.T, origin='lower') #, extent=[0,w,0,h])

fg, hg = plt.subplots(1, 2) #, sharex=True, sharey=True, subplot_kw={'aspect':'equal'})
fg.set_size_inches([9.6, 3.5])
my_world.add_global_cost(CombinedGlobalPotential(goal, zeta=1.0e-4, d_star=50.0))
h_potential = hg[0].matshow(my_world.cost_map.T, origin='lower', cmap=cm.coolwarm)  # , extent=[0,w,0,h]
hg[0].xaxis.set_ticks_position('bottom')
hg[0].plot(goal[0], goal[1], 'yo')

X, Y = np.meshgrid(np.arange(w), np.arange(h))
dx, dy = np.gradient(my_world.cost_map)
magD = np.sqrt(dx**2 + dy**2)
magD[magD == 0] =1.0
U = -(dx/magD).T
V = -(dy/magD).T
harr = hg[0].quiver(X[::4,::4], Y[::4,::4], U[::4,::4], V[::4,::4])
h_contour = hg[1].contour(my_world.cost_map.T, np.linspace(0, 0.5, 11), cmap=cm.coolwarm)
plt.colorbar(h_potential, ax=hg[0])
for o in my_world.obs:
    hg[0].add_artist(Circle( o.pos, o.r+o.Q_star, fc='None', ec='k'))
for hh in hg:
    hh.set_xlim(-0.5, w-0.5)
    hh.set_ylim(-0.5, h-0.5)
    hh.set_aspect('equal')

path = gradient_path(dx, dy, start, goal, step=0.1, goal_range=1.0)
h_path = hg[0].plot(path[:, 0], path[:, 1], 'r-')
z_path = np.zeros(path.shape[0])
for i, (xp, yp) in enumerate(path):
    z_path[i] = my_world.cost_map[int(xp), int(yp)]


f3 = plt.figure()
a3 = f3.add_subplot(111, projection='3d')
h_path3 = a3.plot(path[:,0], path[:,1], z_path, 'r-')
h_surf = a3.plot_surface(X, Y, my_world.cost_map.T, cmap=cm.coolwarm)
a3.plot([goal[0]], [goal[1]], [my_world.cost_map[int(goal[0]), int(goal[1])]], 'yo')

angles = np.linspace(-60, 300, 51)[:-1] # A list of 20 angles between 0 and 360

rotator = RotationAnimator(X, Y, my_world.cost_map, goal, start_azim=-60, delta_angle=5.0)
ani = animation.FuncAnimation(rotator.f, rotator.update_anim, 72, interval=10, blit=False)
ani.save('fig/animation.gif', writer='imagemagick', fps=30)
# create an animated gif (20ms between frames)
# rotanimate(a3, angles,'fig/rotating_potentialfield.gif',delay=20)


plt.show(block=False)