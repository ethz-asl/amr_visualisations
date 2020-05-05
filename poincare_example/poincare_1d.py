# Primarily taken from: https://www.math.wisc.edu/~angenent/519.2016s/notes/poincare-map.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


class PoincarePlotter(object):
    # Class for plotting time evolution of one-dimensional periodic dynamic functions (technically any 1D dynamics)
    t = 0.0
    x = []
    artists = []
    y_lim = [-1.0, 1.0]
    legend_entries = []

    def __init__(self, dynamic_function, ax, x0=[0.0], frames=1000, delta_t=1e-2, n_max=1000, graph_max=10.0, fskip=1):
        assert isinstance(dynamic_function, DynamicFunction)
        self.f = dynamic_function
        self.ax = ax[0]
        self.ax2 = ax[1]
        self.delta_t = delta_t
        self.n_max = n_max
        self.x0 = np.array(x0)
        self.graph_max = graph_max
        self.frames = frames
        self.frame_skip = fskip

        self.colours = cm.get_cmap()(np.linspace(0, 1.0, len(self.x0)+2))[1:-1]

        # Solve equations:
        self.x = np.zeros((self.frames + 1, len(self.x0)), dtype=float)
        self.dx_dt = self.x.copy()
        self.x[0] = self.x0

        self.t = np.arange(0.0, (self.frames + 1) * self.delta_t, self.delta_t)

        for i in tqdm(range(self.frames + 1), desc='Solving dynamics: '):

            if i is 0:
                self.dx_dt[i] = self.f.dx_dt(self.x[i], self.t[i])
                continue

            if self.f.xt is not None:
                self.x[i] = self.f.xt(self.t[i], self.x0)
            else:
                # TODO: Change to better integration scheme (RK4)
                # Euler integration
                self.x[i] = self.x[i-1] + self.delta_t * self.dx_dt[i-1]
            self.dx_dt[i] = self.f.dx_dt(self.x[i], self.t[i])

        self.init()

    def init(self):
        self.ax.cla()
        self.ax.set_xlim(0, 2.0)
        self.y_lim = [-1.0, 1.0]
        self.ax2.cla()

        self.artists = []
        self.legend_entries = []
        for x, dx_dt, c in zip(self.x[0], self.dx_dt[0], self.colours):
            l, = self.ax.plot(self.t[0], x, c=c, lw=2.0)
            p, = self.ax.plot(self.t[0], x, 'o', c=l.get_color())
            l2, = self.ax2.plot(x, dx_dt, c=c, lw=2.0)
            p2, = self.ax2.plot(x, dx_dt, 'o', c=l.get_color())
            self.artists.extend([l, p, l2, p2])
            self.legend_entries.append('$x_0={0:0.2f}$'.format(x))
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
        assert i <= self.frames
        i = i*self.frame_skip

        for j in range(self.x.shape[1]):
            self.artists[4*j].set_data(self.t[:i+1], self.x[:i+1, j])               # Lines
            self.artists[4*j+1].set_data(self.t[i], self.x[i, j])               # Points
            self.artists[4*j+2].set_data(self.x[:i+1, j], self.dx_dt[:i+1, j])       # Lines
            self.artists[4*j+3].set_data(self.x[i, j], self.dx_dt[i, j])         # Points

        self.y_lim[0] = max(min(self.y_lim[0], self.x[i].min()-0.1), -self.graph_max)
        self.y_lim[1] = min(max(self.y_lim[1], self.x[i].max()+0.1), self.graph_max)
        self.ax.set_ylim(self.y_lim)
        self.ax.set_xlim(0, max(2.0, self.t[i]))

        y_lim2 = self.ax2.get_ylim()
        x_lim2 = self.ax2.get_xlim()
        x_lim2 = [max(min(x_lim2[0], self.x[i].min()), -self.graph_max),
                  min(max(x_lim2[1], self.x[i].max()), self.graph_max)]
        y_lim2 = [max(min(y_lim2[0], self.dx_dt[i].min()), -self.graph_max),
                  min(max(y_lim2[1], self.dx_dt[i].max()), self.graph_max)]
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
    return - x + 2*np.cos(t)


def linear_exact(t, x0=0):
    return np.exp(-t)*(x0-1) + np.cos(t) + np.sin(t)


def logistic_periodic(x, t):
    return -x*(1+x) + 2*np.cos(2*np.pi*t)


if __name__ == '__main__':
    plt.rc('text', usetex=True)

    linear = DynamicFunction(linear_damper, xt=linear_exact, fname='$\dot{x} = -x + 2\cos(t)$')
    logistic = DynamicFunction(logistic_periodic, fname='$\dot{x} = -x(x+1) + 2\cos(2 \pi t)$', period=1.0)

    delta_t = 0.015
    fskip = 1

    # x_0 = [1.0]
    # x_0 = [0.9, 1.0, 1.1]
    # x_0 = [-9.0, -4.0, -1.0, 0.5, 1.0, 1.5, 3.0, 11.0]
    # x_0 = [-5.0, 0.0, 1.0, 2.0, 7.0]
    # x_0 = np.linspace(-2.0, 2.0, 5)
    # x_0 = [0.0]
    # x_0 = [-1.0, 0.0, 1.0, 2.0]
    x_0 = np.linspace(-1.1, -1.0, 11)
    # x_0 = np.append(x_0, [0.0, 1.0, 5.0, 10.0])

    with plt.style.context('ggplot'):
        fh, ah = plt.subplots(1, 2)
        fh.set_size_inches([13.5, 6])   #[8.5, 4])   #

        animator = PoincarePlotter(logistic, ah, x0=x_0, delta_t=delta_t, fskip=fskip)

        animation = FuncAnimation(fh, animator.animate, init_func=animator.init, frames=int(1000/fskip), interval=20*fskip, blit=True)
        # animation.save('vid/poincare_example.mp4', writer='ffmpeg', dpi=200, fps=1000.0/(20*fskip),
        #                extra_args=["-crf", "18", "-profile:v", "main", "-tune", "animation", "-pix_fmt", "yuv420p"])
        # animation.save('vid/poincare_example.gif', writer='imagemagick', fps=1000.0/(20*fskip))
        plt.show()
