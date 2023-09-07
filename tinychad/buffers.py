from __future__ import annotations
import numpy as np 
from tinychad.ops_type import UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS

op_map = { 
    BinaryOPS.ADD   : lambda x, y: np.add(x,y),
    BinaryOPS.SUB   : lambda x, y: np.subtract(x,y),
    BinaryOPS.MUL   : lambda x, y: np.multiply(x,y),
    BinaryOPS.DIV   : lambda x, y: np.divide(x,y),
    BinaryOPS.MATMUL: lambda x, y: np.matmul(x,y),

    UnaryOPS.RELU   : lambda x : np.maximum(x, 0),
    UnaryOPS.EXP    : lambda x : np.exp(x),
    UnaryOPS.LOG    : lambda x : np.log(x),
    UnaryOPS.NEG    : lambda x : np.negative(x),
    UnaryOPS.SQRT   : lambda x : np.sqrt(x),

    ShapeOPS.MAX    : lambda x, axis, keepdims : np.max(x, axis, keepdims=keepdims),
    ShapeOPS.SUM    : lambda x, axis, keepdims : np.sum(x, axis, keepdims=keepdims), 

    ReshapeOPS.CAST : lambda x, args : np.broadcast_to(x, args),
    ReshapeOPS.PAD  : lambda x, args : np.pad(x, pad_width = args, mode = 'constant'), 
    ReshapeOPS.TRANSPOSE : lambda x, args : np.transpose(x, args),
    ReshapeOPS.RESHAPE : lambda x, args : np.reshape(x, args)
}

class Buffer: 
    __slots__ = "dat", "shape"
    def __init__(self, dat):
        if isinstance(dat, np.ndarray):
            self.dat = dat

        if isinstance(dat, (int, float, np.float32)):
            self.dat = np.array([dat], dtype=np.float32)

        if isinstance(dat, list):
            self.dat = np.array(dat, dtype=np.float32)
        
        self.shape = self.dat.shape

    def __repr__(self): return f"{self.shape} Buffer"

    @property 
    def dtype(self): return self.dat.dtype

    def __add__(self, x): return self.binary_op(BinaryOPS.ADD, x)
    def __sub__(self, x): return self.binary_op(BinaryOPS.SUB, x)
    def __mul__(self, x): return self.binary_op(BinaryOPS.MUL, x)
    def __div__(self, x): return self.binary_op(BinaryOPS.DIV, x)
    def __matmul__(self, x): return self.binary_op(BinaryOPS.MATMUL, x)

    def exp(self): return self.unary_op(UnaryOPS.EXP)
    def log(self): return self.unary_op(UnaryOPS.LOG)
    def neg(self): return self.unary_op(UnaryOPS.NEG)
    def relu(self): return self.unary_op(UnaryOPS.RELU)
    def sqrt(self): return self.unary_op(UnaryOPS.SQRT)

    # fxn example: BinaryOPS.ADD
    def binary_op(self, fxn, x): return op_map[fxn](self.dat, x.dat)
    def unary_op(self, fxn): return op_map[fxn](self.dat)
    def shape_op(self, fxn, axis, keepdim): return op_map[fxn](self.dat, axis=axis, keepdims=keepdim)
    def reshape_op(self, fxn, args): 
        if fxn == ReshapeOPS.PAD: assert isinstance(args, (tuple, list))
        if fxn == ReshapeOPS.SLICE: return self.slice(args)
        return op_map[fxn](self.dat, args)
        
    def slice(x, args): 
        args = (args) if isinstance(args, int) else args
        out = x.dat[tuple(*args)]
        return out if out.shape != () else [out]


# new lazy buffer
class LazyBuffer: 
    def __init__(self, shape, op, children): 
        pass

# this may just be a LazyBuffer with a different codegen module
class GPUBuffer: 
    def __init__(self, shape, op):
        pass
