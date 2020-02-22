import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import poincare_1d

rc('text', usetex=True)


def calculate_poincare_map(dyn_func, x0, nt=100):

    if dyn_func.xt is not None:     # Have exact solution
        Px = dyn_func.xt(dyn_func.period, x0)
    else:
        Px = np.array(x0).copy()
        dt = dyn_func.period/nt
        tt = np.arange(0.0, dyn_func.period, dt)
        # TODO: Use a better integration scheme (RK4)
        for t in tt:
            dxdt = dyn_func.dx_dt(Px, t)
            Px = Px + dxdt * dt

    return Px


x_0 = np.linspace(-1.2, 2.0, 201)

linear = poincare_1d.DynamicFunction(poincare_1d.linear_damper, xt=poincare_1d.linear_exact, fname='$\dot{x} = -x + 2\cos(t)$')
logistic = poincare_1d.DynamicFunction(poincare_1d.logistic_periodic, fname='$\dot{x} = -x(x+1) + 2\cos(2 \pi t)$', period=1.0)

Px = calculate_poincare_map(logistic, x_0)

with plt.style.context('ggplot'):
    fh, ah = plt.subplots()
    ah.set_xlabel('$x_0$')
    ah.set_ylabel('$P(x_0)$')
    hP = ah.plot(x_0, Px)
    ah.plot([x_0[0], x_0[-1]], [x_0[0], x_0[-1]], '--')
    plt.show()