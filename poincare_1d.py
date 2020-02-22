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

    def __init__(self, dynamic_function, ax, x0=[0.0], delta_t=1e-2, n_max=1000, graph_max=10.0):
        self.f = dynamic_function
        self.ax = ax[0]
        self.ax2 = ax[1]
        self.delta_t = delta_t
        self.n_max = n_max
        self.x0 = np.array(x0)
        self.graph_max = graph_max

        self.colours = cm.get_cmap()(np.linspace(0, 1.0, len(self.x0)+2))[1:-1]
        self.init()

    def init(self):
        self.t = [0.0]
        self.x = [[x] for x in self.x0]
        self.dxdt = [[self.f.dx_dt(x, self.t[0])] for x in self.x0]
        self.ax.cla()
        self.ax.set_xlim(0, 2.0)
        self.y_lim = [-1.0, 1.0]
        self.ax2.cla()

        self.artists = []
        self.legend_entries = []
        for x, dxdt, c in zip(self.x, self.dxdt, self.colours):
            l, = self.ax.plot(self.t, x, c=c, lw=2.0)
            p, = self.ax.plot(self.t, x, 'o', c=l.get_color())
            l2, = self.ax2.plot(x, dxdt, c=c, lw=2.0)
            p2, = self.ax2.plot(x[0], dxdt[0], 'o', c=l.get_color())
            self.artists.extend([l, p, l2, p2])
            self.legend_entries.append('$x_0={0:0.2f}$'.format(x[0]))
        self.ax.grid(True)
        self.ax2.grid(True)
        self.ax.set_xlabel('$t$')
        self.ax.set_ylabel('$x$')
        self.ax.set_title(self.f.fname)
        self.ax2.set_xlabel('$x$')
        self.ax2.set_ylabel('$\partial x / \partial t$')
        self.ax2.set_title(self.f.fname)

        if len(self.x0) <= 11:
            self.ax.legend(self.artists[0:-1:4], self.legend_entries, loc=4)
        return self.artists

    def animate(self, i):
        self.t.append(self.t[-1] + self.delta_t)
        self.t = self.t[-max(self.n_max, i):]
        y_lim2 = self.ax2.get_ylim()
        x_lim2 = self.ax2.get_xlim()
        for i, (x, dxdt, x0) in enumerate(zip(self.x, self.dxdt, self.x0)):
            if self.f.xt is not None:
                nx = self.f.xt(self.t[-1], x0)
            else:
                nx = x[-1] + self.delta_t*dxdt[-1]
            self.y_lim[0] = max(min(self.y_lim[0], nx-0.1), -self.graph_max)
            self.y_lim[1] = min(max(self.y_lim[1], nx+0.1), self.graph_max)
            x.append(nx)
            ndxdt = self.f.dx_dt(x[-1], self.t[-2])
            dxdt.append(ndxdt)
            x = x[-max(self.n_max, i):]
            dxdt = dxdt[-max(self.n_max, i):]

            self.artists[4*i].set_data(np.array(self.t), x)                  # Lines
            self.artists[4*i+1].set_data(self.t[-1], x[-1])        # Points
            self.artists[4*i+2].set_data(x, dxdt)                  # Lines
            self.artists[4*i+3].set_data(x[-1], dxdt[-1])        # Points

            x_lim2 = [max(min(x_lim2[0], x[-1]), -self.graph_max), min(max(x_lim2[1], x[-1]), self.graph_max)]
            y_lim2 = [max(min(y_lim2[0], dxdt[-1]), -self.graph_max), min(max(y_lim2[1], dxdt[-1]), self.graph_max)]
        self.ax.set_ylim(self.y_lim)
        self.ax.set_xlim(0, max(2.0, self.t[-1]))
        self.ax2.set_xlim(x_lim2)
        self.ax2.set_ylim(y_lim2)

        return self.artists


class DynamicFunction(object):
    def __init__(self, dx_dt, xt=None, fname='', period=2*np.pi):
        self.dx_dt = dx_dt
        self.xt = xt        # Exact solution for x(t) (if known)
        self.fname = fname
        self.period = period


def linear_damper(x, t):
    x_dot = - x + 2*np.cos(t)
    return x_dot


def linear_exact(t, x0):
    return np.exp(-t)*(x0-1) + np.cos(t) + np.sin(t)


def logistic_periodic(x, t):
    return -x*(1+x) + 2*np.cos(2*np.pi*t)


if __name__ == '__main__':
    linear = DynamicFunction(linear_damper, xt=linear_exact, fname='$\dot{x} = -x + 2\cos(t)$')
    logistic = DynamicFunction(logistic_periodic, fname='$\dot{x} = -x(x+1) + 2\cos(2 \pi t)$', period=1.0)

    delta_t = 0.015

    # x0 = [1.0]
    # x0 = [0.9, 1.0, 1.1]
    # x0 = [-9.0, -4.0, -1.0, 0.5, 1.0, 1.5, 3.0, 11.0]
    # x0 = [-5.0, 0.0, 1.0, 2.0, 7.0]
    # x0 = np.linspace(-2.0, 2.0, 5)
    # x0 = [0.0]
    # x0 = [-1.0, 0.0, 1.0, 2.0]
    x0 = np.linspace(-1.1, -1.0, 11)
    # x0 = np.append(x0, [0.0, 1.0, 5.0, 10.0])

    with plt.style.context('ggplot'):
        fh, ah = plt.subplots(1, 2)
        fh.set_size_inches([13.5, 6])

        animator = PoincarePlotter(logistic, ah, x0=x0, delta_t=delta_t)

        animation = FuncAnimation(fh, animator.animate, init_func=animator.init, frames=1000, interval=20)
        # animation.save('vid/single_x0.mp4', writer='ffmpeg')
        plt.show()
