# Primarily taken from: https://www.math.wisc.edu/~angenent/519.2016s/notes/poincare-map.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
from matplotlib.animation import FuncAnimation


## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

class PoincarePlotter(object):
    t = 0.0
    x = []
    artists = []

    def __init__(self, dx_dt, ax, n_samples=1, delta_t=1e-2, n_max=1000):
        self.dx_dt = dx_dt
        self.ax = ax
        self.delta_t = delta_t
        self.n_max = n_max

        if n_samples == 1:
            self.x0 = [0.0]
        else:
            self.x0 = np.linspace(-2.0, 2.0, n_samples)

        self.colours = cm.viridis(np.linspace(0, 1.0, n_samples))
        self.init()

    def init(self):
        self.t = [0.0]
        self.x = [[x] for x in self.x0]
        self.ax.cla()
        self.ax.set_xlim(0, 1.0)
        self.y_lim = [-1.0, 1.0]
        self.artists = []
        for x, c in zip(self.x, self.colours):
            l, = self.ax.plot(self.t, x, color=c)
            p, = self.ax.plot(self.t, x, 'o', color=c)
            self.artists.append(l)
            self.artists.append(p)
        self.ax.grid()
        self.ax.set_xlabel('$t$')
        self.ax.set_ylabel('$x$')
        return self.artists

    def animate(self, i):
        self.t.append(self.t[-1] + self.delta_t)
        self.t = self.t[-max(self.n_max, i):]
        for i, x in enumerate(self.x):
            nx = x[-1] + self.delta_t*self.dx_dt(x[-1], self.t[-2])
            self.y_lim[0] = min(self.y_lim[0], nx)
            self.y_lim[1] = max(self.y_lim[1], nx)
            x.append(nx)
            x = x[-max(self.n_max, i):]

            self.artists[2*i].set_data(self.t, x)                  # Lines
            self.artists[2*i+1].set_data(self.t[-1], x[-1])        # Points
        self.ax.set_ylim(self.y_lim)
        self.ax.set_xlim(0, max(1.0, self.t[-1]))
        return self.artists


def linear_damper(x, t, k=2.0):
    x_dot = -x + 2*np.cos(t*2*np.pi)
    return x_dot


n_samples = 11
delta_t = 1e-3

fh, ah = plt.subplots()

animator = PoincarePlotter(linear_damper, ah, n_samples=n_samples)

animation = FuncAnimation(fh, animator.animate, init_func=animator.init, frames=1000, interval=20)
plt.show()