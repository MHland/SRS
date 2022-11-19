import numpy as np
from numba import jit
import warnings

class JitIndex:
    jit_index = True
    @classmethod
    def set_jit_index(cls, new_value):
        if isinstance(new_value, bool):
            cls.jit_index = new_value
        else:
            warnings.warn('vectorization is a BOOL value! ')
    @classmethod
    def get_jit_index(cls):
        return cls.jit_index

class MuskingumMethod:
    __jit_index = JitIndex.jit_index
    def __init__(self):
        pass

    @staticmethod
    @jit(nopython= __jit_index)
    def route_simulate_vector_method(index, Qsim, KE, XE, fip, fop, time=24):
        dt = np.ones((index))

        rtot = Qsim
        fin = rtot / time
        fdr = np.zeros((fin.shape))

        c0 = (-2.0 * KE * XE + dt) / (2.0 * KE * (1.0 - XE) + dt)
        c1 = (2.0 * KE * XE + dt) / (2.0 * KE * (1.0 - XE) + dt)
        c2 = (2.0 * KE * (1.0 - XE) - dt) / (2.0 * KE * (1.0 - XE) + dt)

        for i in range(time):
            fon = c0 * fin + c1 * fip + c2 * fop
            fop[fop < 0] = 0  # to set up the value below 0 as 1.0
            fdr = fdr + fon
            fop = fon
            fip = fin
        Qsim_out = fdr
        return Qsim_out, fip, fop

    @staticmethod
    @jit(nopython=__jit_index)
    def route_simulates_scalar_method(Qsim, KE, XE, fip, fop, time=24):
        dt = 1.0

        rtot = Qsim
        fin = rtot / time
        fdr = 0.0

        c0 = (-2.0 * KE * XE + dt) / (2.0 * KE * (1.0 - XE) + dt)
        c1 = (2.0 * KE * XE + dt) / (2.0 * KE * (1.0 - XE) + dt)
        c2 = (2.0 * KE * (1.0 - XE) - dt) / (2.0 * KE * (1.0 - XE) + dt)

        for i in range(time):
            fon = c0 * fin + c1 * fip + c2 * fop
            if fon < 0:
                fon = 0.0  # to set up the value below 0 as 1.0
            fdr = fdr + fon
            fop = fon
            fip = fin
        Qsim_out = fdr
        return Qsim_out, fip, fop