import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean, cdist
import timeit
import time

def kdtree_nn(point_set, target_point):
    tree = KDTree(point_set)
    dist, ind = tree.query(target_point, k=1)
    return dist, ind

def bruteforce_nn(point_set, target_point):
    d_best = euclidean(point_set[0], target_point)
    i_best = 0
    for i, p in enumerate(point_set[1:]):
        d = euclidean(p, target_point)
        if d < d_best:
            d_best = d
            i_best = i+1
    return d_best, i_best

def cdist_nn(point_set, target_point):
    d_full = cdist(point_set, [target_point]).flatten()
    i_best = np.argmin(d_full)
    return d_full[i_best], i_best

def wrapper(func, x_rand):
    def wrapped():
        func(x_rand[1:], x_rand[0])
    return wrapped

# Value test
x_rand = np.random.uniform([0, 0], [1, 1], size=(1000 + 1, 2))
for func in [kdtree_nn, cdist_nn, bruteforce_nn]:
    d, i = func(x_rand[1:], x_rand[0])
    print('{0}: d={1}, i={2}'.format(func.__name__, d, i))

for n_points in [10, 1000, 100000]:
    for func in [kdtree_nn, cdist_nn, bruteforce_nn]:
        np.random.seed(1)
        x_rand = np.random.uniform([0, 0], [1, 1], size=(n_points + 1, 2))
        wrapped = wrapper(func, x_rand)
        number = int(1000000/n_points)
        t = timeit.timeit(wrapped, number=number)
        print('{0} (n={1}): {2}'.format(func.__name__, n_points, t))

# cdist seems insanely fast... just do a second sanity check
def time_test(func, n_points):
    x_rand = np.random.uniform([0, 0], [1, 1], size=(n_points + 1, 2))
    return func(x_rand[1:], x_rand[0])

for func in [kdtree_nn, cdist_nn, bruteforce_nn]:
    n_points = 10000
    t0 = time.time()
    for i in range(1000):
        time_test(func, n_points)
    print('{0} (n={1}): {2}'.format(func.__name__, n_points, time.time()-t0))