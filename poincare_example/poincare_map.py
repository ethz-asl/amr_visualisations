import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from poincare_1d import DynamicFunction, linear_damper, linear_exact, logistic_periodic

rc('text', usetex=True)


def calculate_poincare_map(dyn_func, x0, nt=100):
    # Calculate the Poincar\'e map for a given dynamic function
    # If exact solution known, solve directly. If not, Euler integration with nt steps
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


def follow_stability(x, Px, x0, n_jumps=5):
    xx = [x0]
    for i in range(n_jumps):
        xx.append(np.interp(xx[-1], Px, x))
    return xx


x_0 = np.linspace(-3.0, 5.0, 201)

linear = DynamicFunction(linear_damper, xt=linear_exact, fname='$\dot{x} = -x + 2\cos(t)$')
logistic = DynamicFunction(logistic_periodic, fname='$\dot{x} = -x(x+1) + 2\cos(2 \pi t)$', period=1.0)

test_fun = linear
Px = calculate_poincare_map(test_fun, x_0)

with plt.style.context('ggplot'):
    fh, ah = plt.subplots()
    ah.set_xlabel('$x_0$')
    ah.set_ylabel('$P(x_0)$')
    hP, = ah.plot(x_0, Px)
    h0, = ah.plot([x_0[0], x_0[-1]], [x_0[0], x_0[-1]], '--')
    ah.set_title(r"Poincar\'e map for {0}".format(test_fun.fname))
    ah.legend([hP, h0], ['$P(x_0)$', '$P(x)=x$'])
    plt.show()
