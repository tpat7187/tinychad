from __future__ import annotations
import numpy as np 
from tinychad.ops_type import UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS

class Buffer: 
    def __init__(self, dat):
        if isinstance(dat, np.ndarray):
            self.dat = dat

        if isinstance(dat, (int, float, np.float32)):
            self.dat = np.array([dat], dtype=np.float32)

        if isinstance(dat, list):
            self.dat = np.array(dat, dtype=np.float32)
        

    def __repr__(self): return str(self.dat)

    @property 
    def shape(self): return self.dat.shape

    @property 
    def dtype(self): return self.dat.dtype

    # fxn example: BinaryOPS.ADD
    def binary_op(self, fxn, x): 
        return fxn(self.dat, x.dat)

    def unary_op(self, fxn):
        if fxn == UnaryOPS.RELU: return np.maximum(self.dat, 0)
        else: return fxn(self.dat)

    def matmul(self, x):
        return np.matmul(self.dat, x.dat)

    def shape_op(self, fxn, axis, keepdim):
        return fxn(self.dat, axis=axis, keepdims=keepdim)


# lazy buffer
class CompiledBuffer: 
    def __init__(self, shape, op, children): 
        pass
