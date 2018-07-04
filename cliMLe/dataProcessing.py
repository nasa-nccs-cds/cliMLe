import os, sys, math
import cdms2 as cdms
import numpy as np

class PreProc:
    smoothing_kernel = np.array([.13, .23, .28, .23, .13])

    @staticmethod
    def normalize( data, axis=0 ):
        # type: (np.ndarray) -> np.ndarray
        std = np.std( data, axis )
        return data / std

    @classmethod
    def lowpass( cls, data ):
        # type: (np.ndarray) -> np.ndarray
        return np.convolve( data, cls.smoothing_kernel, "same")