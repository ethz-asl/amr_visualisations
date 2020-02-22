import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

x0 = np.linspace(0, 2, 201)
Px = np.exp(-2*np.pi)*(x0-1) + 1

with plt.style.context('ggplot'):
    fh, ah = plt.subplots()
    ah.set_xlabel('$x_0$')
    ah.set_ylabel('$P(x)$')
    hP = ah.plot(x0, Px)
    ah.plot([x0[0], x0[-1]], [x0[0], x0[-1]], '--')
    plt.show()