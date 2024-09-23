from __future__ import annotations
import numpy as np
from tinychad.tensor import OP
from typing import Union, Optional
from tinychad.buffers import Buffer
from tinychad.ops_type import UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS

# TODO: at some point we need to make the backward pass work again for now we just care about the actual compiler part
 
# binary ops
class ADD(OP): 
  __slots__ = "x", "y"
  @classmethod
  def forward(self, x:Buffer, y:Buffer) -> Buffer: 
    self.x, self.y = x, y 
    return x.binary_op(BinaryOPS.ADD, y)
  
  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return out_grad, out_grad

class SUB(OP): 
  __slots__ = "x", "y"
  @classmethod
  def forward(self, x:Buffer, y:Buffer) -> Buffer:
    self.x, self.y = x, y 
    return x.binary_op(BinaryOPS.SUB, y)
  
  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return out_grad, -out_grad

class MUL(OP): 
  __slots__ = "x", "y"
  @classmethod
  def forward(self, x:Buffer, y:Buffer) -> Buffer:
    self.x, self.y = x, y 
    return x.binary_op(BinaryOPS.MUL, y)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return out_grad * self.y.data, out_grad*self.x.data

class DIV(OP): 
  __slots__ = "x", "y"
  @classmethod
  def forward(self, x:Buffer, y:Buffer) -> Buffer:
    self.x, self.y = x, y 
    return x.binary_op(BinaryOPS.DIV, y)
  
  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return (self.y.data()**-1) * out_grad, -(self.x.data()/self.y.data()**2)*out_grad

class MATMUL(OP): 
  __slots__ = "x", "y"
  @classmethod
  def forward(self, x:Buffer, y:Buffer) -> Buffer:
    self.x, self.y = x, y 
    return x.binary_op(BinaryOPS.MATMUL, y)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return np.matmul(out_grad, self.y.data.T), np.matmul(self.x.data.T, out_grad)

class GTT(OP): 
  __slots__ = "x", "y" 
  @classmethod
  def forward(self, x: Buffer, y:Buffer) -> Buffer:
    self.x, self.y = x, y 
    return x.binary_op(BinaryOPS.MAX, y)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return out, out

# unary ops
class RELU(OP):
  __slots__ = "x"
  @classmethod
  def forward(self, x:Buffer) -> Buffer:
    self.x = x
    return x.unary_op(UnaryOPS.RELU)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return (out > 0)*out_grad

class EXP(OP): 
  __slots__ = "x"
  @classmethod
  def forward(self, x:Buffer) -> Buffer:
    self.x = x
    return x.unary_op(UnaryOPS.EXP)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return out * out_grad

class LOG(OP): 
  __slots__ = "x"
  @classmethod
  def forward(self, x:Buffer) -> Buffer:
    self.x = x
    return x.unary_op(UnaryOPS.LOG)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return out_grad / self.x.data

class NEG(OP): 
  __slots__ = "x"
  @classmethod
  def forward(self, x:Buffer) -> Buffer:
    self.x = x
    return x.unary_op(UnaryOPS.NEG)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return -1*out_grad

class SQRT(OP): 
  __slots__ = "x"
  @classmethod
  def forward(self, x:Buffer) -> Buffer:
    self.x = x
    return x.unary_op(UnaryOPS.SQRT)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return (1 / 2 * out_grad**2)

# shape ops
class SUM(OP):
  __slots__ = "x", "axis", "keepdim"
  @classmethod
  def forward(self, x:Buffer, axis:Optional[int], keepdim:bool) -> Buffer:
    self.x, self.axis, self.keepdim = x, axis, keepdim
    return x.shape_op(ShapeOPS.SUM, axis, keepdim)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    print(self.x)
    return np.broadcast_to(out_grad, self.x.shape)
      
class MAX(OP): 
  __slots__ = "x", "axis", "keepdim"
  @classmethod
  def forward(self, x:Buffer, axis:Optional[int], keepdim:bool) -> Buffer: 
    self.x, self.axis, self.keepdim = x, axis, keepdim
    return x.shape_op(ShapeOPS.MAX, axis, keepdim)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    if self.keepdim is False:
      out = np.expand_dims(out, axis=self.axis) if self.axis is not None else out
      out_grad = np.expand_dims(out_grad, axis=self.axis) if self.axis is not None else out_grad
    tt = 1.0 - (self.x.data < np.broadcast_to(out, self.x.shape)).astype(np.float32)
    exp = np.broadcast_to(tt.sum(axis=self.axis,keepdims=True), self.saved[0].shape)
    out = (tt / exp) * np.broadcast_to(out_grad, self.saved[0].shape)
    return out

# reshape ops
class RESHAPE(OP): 
  __slots__ = "x", "args"
  @classmethod
  def forward(self, x:Buffer, args) -> Buffer:
    self.x, self.args = x, args
    return x.reshape_op(ReshapeOPS.RESHAPE, args)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return out_grad.reshape(self.saved[0].shape)

class CAST(OP):
  __slots__ = "x", "args"
  @classmethod
  def forward(self, x:Buffer, args) -> Buffer:
    self.x, self.args = x, args
    return x.reshape_op(ReshapeOPS.CAST, args)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    diff = len(out_grad.shape) - len(self.x.shape)
    if diff > 0: out_grad = out_grad.sum(axis=tuple(np.arange(diff)))
    t = tuple([i for i, (a, b) in enumerate(zip(out_grad.shape, self.x.shape)) if a != b])
    out_grad = out_grad.sum(axis = t, keepdims = True)
    return out_grad

class SLICE(OP):
  __slots__ = "x", "args"
  @classmethod
  def forward(self, x:Buffer, args) -> Buffer:
    self.x, self.args = x, args
    return x.reshape_op(ReshapeOPS.SLICE, args)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    arg = self.ctx[0]
    acc = np.zeros_like(self.x.data)
    np.add.at(acc, *arg, out_grad)
    return acc

class PAD(OP): 
  __slots__ = "x", "args"
  @classmethod
  def forward(self, x: Buffer, args) -> Buffer:
    self.x, self.args = x, args
    return x.reshape_op(ReshapeOPS.PAD, args)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    w = tuple([slice(i[0], j-i[1], None) for i, j in zip(*self.ctx, out.shape)])
    out = out_grad[w]
    return out

class TRANSPOSE(OP): 
  __slots__ = "x", "args"
  @classmethod
  def forward(self, x:Buffer, args) -> Buffer:
    self.x, self.args = x, args
    return x.reshape_op(ReshapeOPS.TRANSPOSE, args)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return np.transpose(out_grad, np.argsort(*self.ctx))

 


