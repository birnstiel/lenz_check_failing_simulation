from dustpy.sim.constants import AU
import numpy as np


def smoothstep(x, w):
    """
    This file produces a "smoothed heaviside" function

    y = 1/2*exp(x/w)       if x <= 0
    y = 1-1/2*exp(-x/w)    if x >  0

    Arguments
    ---------
    x : array-like
        input x-array

    w : float
        width of the transition

    Output
    ------
    y : array-like
        the function at every given x value

    """
    nd = np.ndim(x)
    x = np.array(x, ndmin=1)
    y = np.zeros(np.size(x))

    for i, r in enumerate(x):
        if r <= 0.:
            y[i] = 0.5 * np.exp(r / w)
        else:
            y[i] = 1. - 0.5 * np.exp(-r / w)
    if nd == 0:
        return y[0]
    else:
        return y


def smooth_vfrag(sim):
    i_sl = np.abs(sim.gas.T - 170.).argmin()
    r_sl = np.interp(170, sim.gas.T[i_sl - 1:i_sl + 2], sim.grid.r[i_sl - 1:i_sl + 2])
    return 100. * 10.**smoothstep(sim.grid.r - r_sl, 0.25 * AU)
