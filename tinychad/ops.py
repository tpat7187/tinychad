import numpy as np
from tinychad.tensor import OP
from typing import Union
from enum import Enum, auto

class UnaryOPS(Enum): RELU = auto(); NEG = auto(); LOG = auto(); EXP = auto(); SQRT = auto();
class BinaryOPS(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MATMUL = auto(); 
class ShapeOPS(Enum): MAX = auto(); SUM = auto();
class ReshapeOPS(Enum): RESHAPE = auto(); SLICE = auto(); PAD = auto(); TRANSPOSE = auto();


class LOAD(OP): 
  def __init__(self, saved = None):
    self.arg = type(self).__name__
    self.saved = saved

# binary ops
class ADD(OP): 
  @staticmethod
  def forward(x, y): return x.data + y.data
  
  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad
    self.saved[1].grad += out_grad

class SUB(OP): 
  @staticmethod
  def forward(x, y): return x.data - y.data
  
  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad
    self.saved[1].grad += -out_grad

class MUL(OP): 
  @staticmethod
  def forward(x, y): return x.data * y.data

  def backward(self, out_grad, out):
    self.saved[0].grad += self.saved[1].data * out_grad
    self.saved[1].grad += self.saved[0].data * out_grad

class DIV(OP): 
  @staticmethod
  def forward(x, y): return x.data / y.data
  
  def backward(self, out_grad, out):
    self.saved[0].grad += (self.saved[1].data**-1) * out_grad
    self.saved[1].grad += -(self.saved[0].data/self.saved[1].data**2) * out_grad

class MATMUL(OP): 
  @staticmethod
  def forward(x, y): return np.matmul(x.data, y.data)

  def backward(self, out_grad, out):
    self.saved[0].grad += np.matmul(out_grad, self.saved[1].T().data)
    self.saved[1].grad += np.matmul(self.saved[0].T().data, out_grad)

# unary ops
class RELU(OP):
  @staticmethod
  def forward(x): return np.maximum(x.data, 0)

  def backward(self, out_grad, out):
    self.saved[0].grad += (out > 0) *out_grad

class EXP(OP): 
  @staticmethod
  def forward(x): return np.exp(x.data)

  def backward(self, out_grad, out):
    self.saved[0].grad += out * out_grad

class LOG(OP): 
  @staticmethod
  def forward(x): return np.log(x.data)

  def backward(self, out_grad, out):
    self.saved[0].grad += out_grad / self.saved[0].data

class NEG(OP): 
  @staticmethod
  def forward(x): return -1*x.data

  def backward(self, out_grad, out): 
    self.saved[0].grad += -1*out_grad

class SQRT(OP): 
  @staticmethod 
  def forward(x): return np.sqrt(x.data)

  def backward(self, out_grad, out): 
    self.saved[0].grad += (1 / 2 * out_grad**2)

# shape ops
class SUM(OP):
  @staticmethod
  def forward(x, axis, keepdim):
    return np.array([x.data.sum(keepdims = keepdim)]) if axis is None else x.data.sum(axis=axis, keepdims = keepdim)

  def backward(self, out_grad, out):
    if not isinstance(self.ctx, int):
      self.saved[0].grad += out_grad 
    else: 
      self.saved[0].grad += np.broadcast_to(out_grad, self.saved[0].grad.shape)
      
class MAX(OP): 
  @staticmethod
  def forward(x, axis, keepdim): 
    return np.array([x.data.max(keepdims = keepdim)]) if axis is None else x.data.max(axis=axis, keepdims = keepdim)

  def backward(self, out_grad, out):
    axis, kd = self.ctx[0], self.ctx[1]
    out_t = np.expand_dims(out, axis=axis) if axis is not None else out
    out_grad = np.expand_dims(out_grad, axis=axis) if axis is not None else out_grad
    tt = 1.0 - (self.saved[0].data < np.broadcast_to(out_t, self.saved[0].shape))
    exp = np.broadcast_to(tt.sum(axis=axis,keepdims=True), self.saved[0].shape)
    self.saved[0].grad += (tt / exp) * np.broadcast_to(out_grad, self.saved[0].shape)

# reshape ops
class RESHAPE(OP): 
  @staticmethod 
  def forward(x, args): return np.reshape(x.data, args)

  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad.reshape(self.saved[0].shape)

class CAST(OP):
  @staticmethod 
  def forward(x, args): return np.broadcast_to(x.data, args)

  def backward(self, out_grad, out): 
    diff = len(out_grad.shape) - len(self.saved[0].shape)
    if diff > 0: out_grad = out_grad.sum(axis=tuple(np.arange(diff)))
    t = tuple([i for i, (a, b) in enumerate(zip(out_grad.shape, self.saved[0].shape)) if a != b])
    out_grad = out_grad.sum(axis = t, keepdims = True)
    self.saved[0].grad += out_grad

class SLICE(OP):
  @staticmethod
  def forward(x, args): 
    out = x.data[tuple(*args)]
    return out if out.shape != () else [out]

  def backward(self, out_grad, out):
    arg = self.ctx[0]
    # accumulate gradients ; validate that this works for small slices?
    acc = np.zeros_like(self.saved[0].grad)
    np.add.at(acc, *arg, out_grad)
    self.saved[0].grad += acc

# the ctx in the backward pass is passed in with *args, because of the new generic reshape_op
class PAD(OP): 
  @staticmethod 
  def forward(x, args):
    assert isinstance(args, (tuple, list))
    out = np.pad(x.data, pad_width=args, mode='constant')
    return out

  def backward(self, out_grad, out): 
    w = tuple([slice(i[0], j-i[1], None) for i, j in zip(*self.ctx, out.shape)])
    self.saved[0].grad += out_grad[w]

class TRANSPOSE(OP): 
  @staticmethod
  def forward(x, args): return np.transpose(x.data, args)

  def backward(self, out_grad, out): 
    self.saved[0].grad += np.transpose(out_grad, np.argsort(*self.ctx))

 


