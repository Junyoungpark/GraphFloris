# Windows doesn't work with 'signal' package, so implement using multiprocessing
from multiprocessing import Process, Queue

import numpy as np


def timeout(seconds=5, action=None):
    """Calls any function with timeout after 'seconds'.
       If a timeout occurs, 'action' will be returned or called if
       it is a function-like object.
    """

    def handler(queue, func, args, kwargs):
        queue.put(func(*args, **kwargs))

    def decorator(func):

        def wraps(*args, **kwargs):
            q = Queue()
            p = Process(target=handler, args=(q, func, args, kwargs))
            p.start()
            p.join(timeout=seconds)
            if p.is_alive():
                p.terminate()
                p.join()
                if hasattr(action, '__call__'):
                    return action()
                else:
                    return action
            else:
                return q.get()

        return wraps

    return decorator


def valid_layout(x_coords, y_coords, min_dist):
    """
    :param x_coords: (list of floats) list of x coordinates in the grid
    :param y_coords: (list of floats) list of y coordinates in the grid
    :param min_dist: (scalar) minimum distance bet. two arbitrary points to be
                    satisfied by the coordinates.
    :return: validity flag
    """

    # the dumbest way for checking validity of the given layout
    # will be / need to be revised !!

    valid = True
    coords = zip(x_coords, y_coords)
    for c1 in coords:
        for c2 in coords:
            dist = np.linalg.norm(np.array(c1) - np.array(c2))
            if dist <= min_dist:
                valid = False
                return valid
    return valid


def sequential_sampling(x_grid_size, y_grid_size, min_dist, num_turbines, max_retry: int = 500):
    """
    :param x_grid_size: (float) X axis grid size
    :param y_grid_size: (float) Y axis grid size
    :param min_dist: (scalar) minimum distance bet. two arbitrary points to be
                    satisfied by the coordinates.
    :param num_turbines: (int) number of sample turbines
    :return: turbine_x_coords, turbine_y_coords
    """

    # the dumbest way for sampling wind turbine layout
    # Do not guarantee whether given input params are valid for sampling
    # will be / need to be revised

    # Check whether the new coord. is valid against the valid' coords set.
    def valid_sample(x_coords, y_coords, new_x, new_y):
        valid = True
        for x, y in zip(x_coords, y_coords):
            dist = np.linalg.norm(np.array([x, y]) - np.array([new_x, new_y]))
            if dist <= min_dist:
                valid = False
                return valid
        return valid

    @timeout(100)
    def sampling():
        turbine_x_coords = []
        turbine_y_coords = []

        while len(turbine_x_coords) < num_turbines:
            x = np.random.uniform(low=0.0, high=x_grid_size)
            y = np.random.uniform(low=0.0, high=y_grid_size)
            if len(turbine_x_coords) == 0:
                pass
            else:
                if not valid_sample(turbine_x_coords, turbine_y_coords, x, y):
                    continue
            turbine_x_coords.append(x)
            turbine_y_coords.append(y)
            print("len", len(turbine_x_coords))
        return turbine_x_coords, turbine_y_coords

    success = False
    for i in range(max_retry):
        print(i)
        try:
            turbine_x_coords, turbine_y_coords = sampling()
            success = True
        except:
            pass

    if not success:
        raise RuntimeError(
            "sequential sampling cannot find the possible case. retry with lager grids or smaller turbines.")

    return turbine_x_coords, turbine_y_coords
