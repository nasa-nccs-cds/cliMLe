import os, sys, math
import cdms2 as cdms
import numpy as np

class PreProc:

    @staticmethod
    def normalize( data ):
        # type: (np.ndarray) -> np.ndarray
        std = np.std( data, 0 )
        return data / std

    @staticmethod
    def lowpass( data, kernel_size ):
        # type: (np.ndarray) -> np.ndarray
        kernel = np.array( [ .13, .23, .28, .23, .13 ])
        return np.convolve( data, kernel, "same")