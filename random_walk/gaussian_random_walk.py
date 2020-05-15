import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import argparse

""" 

Basic random walk visualisation

Requires: numpy, matplotlib, argparse

Author: Nicholas Lawrance (nicholas.lawrance@mavt.ethz.ch)

"""

plt.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)
cmap = cm.viridis

parser = argparse.ArgumentParser(description='Basic random walk')
parser.add_argument('-n', type=int, default=50, help='Number of total trajectories')
parser.add_argument('-t', type=int, default=100, help='Number of timesteps')
parser.add_argument('--sigma', type=float, default=1.0, help='Standard deviation')
parser.add_argument('--fps', type=int, default=50, help='FPS')
parser.add_argument('--n_show', type=int, default=5, help='Number of single paths to show')
parser.add_argument('-vf', '--video-file', type=str, default='', help='Save animation to file')
args = parser.parse_args()

class WalkAnimator(object):

    def __init__(self, n=50, t=100, n_show=10, sigma=1.0, end_append=100):
        assert n_show <= n
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches([4.8, 2.7])  # 1920*1080 at 200 dpi
        self.n = n
        self.t = t
        self.tt = np.arange(self.t)
        self.n_show = n_show
        self.sigma = sigma
        self.end_append = end_append        # Extra frames on the end (freeze)

        self.paths = np.zeros((self.n, self.t))
        self.paths[:, 1:] = np.random.normal(0, self.sigma**2, (self.n, self.t-1)).cumsum(axis=1)

        self.max_frames = self.t*self.n_show + self.t + self.end_append
        self.h_lines = []


    def init_fig(self):
        self.ax.cla()

        self.ax.set_xlim(0, self.t)
        self.ax.set_ylim(self.paths.min(), self.paths.max())
        self.ax.grid()

        self.ax.set_xlabel(r'$t$')
        self.ax.set_ylabel(r'$x(t)$')
        self.ax.set_title(r'Gaussian random walk, $x_{n+1} = x_n + z$, $z \sim \mathcal{N}(0, '+'{0:0.1f})$'.format(self.sigma))

        self.h_lines = []
        for p in self.paths:
            self.h_lines.append(self.ax.plot([0], p[:1])[0])

        return self.h_lines

    def animate(self, i):
        l_n = i // self.t
        k = i % self.t

        if l_n < self.n_show:
            k = i % self.t
            self.h_lines[l_n].set_data(self.tt[:k+1], self.paths[l_n, :k+1])

        elif i < self.max_frames - self.end_append:
            for p, h in zip(self.paths[self.n_show:], self.h_lines[self.n_show:]):
                h.set_data(self.tt[:k+1], p[:k+1])

        return self.h_lines

walker = WalkAnimator(args.n, args.t, n_show=args.n_show, sigma=args.sigma)
anim = animation.FuncAnimation(walker.fig, walker.animate, init_func=walker.init_fig, frames=walker.max_frames,
                          interval=1000.0/args.fps, blit=True)

if args.video_file:
    # animation.save('fig/arm_config_space_video.gif', writer='imagemagick', fps=1000.0/delta_t)
    anim.save(args.video_file, writer='ffmpeg', fps=args.fps, dpi=200,
                       extra_args=["-crf", "18", "-profile:v", "main", "-tune", "animation", "-pix_fmt", "yuv420p"])
plt.show()