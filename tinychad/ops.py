import numpy as np
from tinychad.tensor import OP

class LOAD(OP): 
  def __init__(self, saved = None):
    self.arg = type(self).__name__
    self.saved = saved

''' 
def unbroadcast(): 
  this function will take in a shape, and if the children of the shape have been broadcasted, it will sum itself along an axis to match the child shape

  DOES PYTORCH DO THIS

'''

# self and out_grad
def _unbr(out_grad, saved):
  def _unbroadcast(out_grad,saved):
    return all((m==n) | (m==1) | (n==1) for n,m in \
    zip(out_grad.shape[::-1], saved.shape[::-1]))
  # if true, out_grad needs to be reshaped to saved
  if _unbroadcast(out_grad, saved): 
    if out_grad.shape == saved.shape:
      return out_grad
    else: 
      out_grad = out_grad.sum(axis=1, keepdims = True)
  return out_grad

# takes in input tensor and output shape
# checks if they're broadcastable
# if they are then does the transformation
  

# binary ops
class ADD(OP): 
  @staticmethod
  def forward(x, y): 
    if(x.shape != y.shape): 
      x.data, y.data = CAST.forward(x,y)
    return x.data + y.data
  
  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad
    self.saved[1].grad += out_grad

class SUB(OP): 
  @staticmethod
  def forward(x, y): 
    return x.data - y.data
  
  def backward(self, out_grad, out): 
    self.saved[0].grad += _unbr(out_grad, self.saved[0])
    self.saved[1].grad += _unbr(out_grad, self.saved[1])

class MATMUL(OP): 
  @staticmethod
  def forward(x, y): 
    return np.dot(x.data, y.data)

  def backward(self, out_grad, out):
    self.saved[0].grad += np.dot(out_grad, self.saved[1].T().data)
    self.saved[1].grad += np.dot(self.saved[0].T().data, out_grad)

class MUL(OP): 
  @staticmethod
  def forward(x, y): 
    return x.data * y.data

  def backward(self, out_grad, out):
    self.saved[0].grad += self.saved[1].data * out_grad
    self.saved[1].grad += self.saved[0].data * out_grad


# do we need DIV
class DIV(OP): 
  @staticmethod
  def forward(x, y): 
    return x.data / y.data
  
  def backward(self, out_grad, out):
    self.saved[0].grad += _unbr((self.saved[1].data**-1) * out_grad, self.saved[0].grad)
    self.saved[1].grad += _unbr(-(self.saved[0].data/self.saved[1].data**2) * out_grad, self.saved[1].grad)

# unary ops
class SUM(OP):
  @staticmethod
  def forward(x, axis, keepdim):
    return np.array([x.data.sum(keepdims = keepdim)]) if axis is None else x.data.sum(axis=axis, keepdims = keepdim)

  def backward(self, out_grad, out):
    if not isinstance(self.ctx, int):
      self.saved[0].grad += out_grad 
    else: 
      self.saved[0].grad += np.broadcast_to(out_grad, self.saved[0].grad.shape)

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
  def forward(x): 
    return np.log(x.data)

  def backward(self, out_grad, out):
    self.saved[0].grad += out_grad / self.saved[0].data

class MAX(OP): 
  @staticmethod
  def forward(x, axis, keepdim): 
    return np.array([x.data.argmax(keepdims = keepdim)]) if axis is None else x.data.argmax(axis=axis, keepdims = keepdim)

  def backward(self, out_grad, out):
    if not isinstance(self.ctx, int):
      self.saved[0].grad += out_grad 
    else: 
      self.saved[0].grad += np.broadcast_to(out_grad, self.saved[0].grad.shape)

# shapes

class RESHAPE(OP): 
  @staticmethod 
  def forward(x, *shape): return np.reshape(x.data, shape)

  def backward(self, out_grad, out): 
    self.saved[0].grad = out_grad.reshape(self.saved[0].shape)


# prett unary i guess
class CAST(OP):
  @staticmethod 
  # return broadcasted x,y
  def forward(x, y):
    out_s = np.broadcast_shapes(x.shape, y.shape)
    return np.broadcast_to(x.data, out_s)

  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad.sum(axis=1,keepdims=True)





def is_castable(x,y):
  return all((m==n) | (m==1) | (n==1) for n,m in \
  zip(x.shape[::-1], y.shape[::-1]))












  













    















