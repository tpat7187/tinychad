import numpy as np
from tinychad.tensor import OP

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
  def forward(x, y): return np.dot(x.data, y.data)

  def backward(self, out_grad, out):
    self.saved[0].grad += np.dot(out_grad, self.saved[1].T().data)
    self.saved[1].grad += np.dot(self.saved[0].T().data, out_grad)

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
    if axis is None: 
      return np.array([x.data.max(keepdims = keepdim)])
    else:
      return x.data.max(axis=axis, keepdims = keepdim)

  def backward(self, out_grad, out):
    axis, kd = self.ctx[0], self.ctx[1]
    # broadcasting 'direction' changes depending on the axis we use
    if axis == 1:
      tt = np.broadcast_to(out.reshape(-1,1), self.saved[0].shape)
    else: 
      tt = np.broadcast_to(out, self.saved[0].shape)
    tt = (self.saved[0].data == tt).astype(np.promote_types(self.saved[0].data.dtype, tt.dtype))
    expand = np.broadcast_to(tt.sum(axis=axis, keepdims = kd), self.saved[0].shape)
    max_amount = tt / expand
    grad_output_exp = np.broadcast_to(out_grad, self.saved[0].shape)
    self.saved[0].grad += max_amount * grad_output_exp

# reshape ops
class RESHAPE(OP): 
  @staticmethod 
  def forward(x, *shape):  return np.reshape(x.data, shape)

  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad.reshape(self.saved[0].shape)

# TODO: once the shapes are equal, we need to find the axis to sum over to make them the same
class CAST(OP):
  @staticmethod 
  def forward(x, y): return np.broadcast_to(x.data, y)

  def backward(self, out_grad, out): 
    # can we do this better? do we need a ctx
    shp, r = self.ctx, out_grad
    for j in range(len(out_grad.shape) - len(self.saved[0].shape)):
      r = r.sum(axis=0)
    if len(r.shape) == len(self.saved[0].shape) and len(r.shape)>1 and len(self.saved[0].shape)>1:
      ss = 0 
      for i,j in zip(r.shape, self.saved[0].shape): 
        if i != j: 
          break
        ss+=1
      r = r.sum(axis=ss, keepdims = True)
    self.saved[0].grad += r

# we support LOCAL slicing [x,y,z] NOT [x][y][z] idk if this is bad 
class SLICE(OP):
  @staticmethod
  def forward(x, args): 
    return x.data[args] if isinstance(args, (slice, tuple)) else np.array([x.data[args]])

  def backward(self, out_grad, out):
    arg = self.ctx[0]
    self.saved[0].grad[arg] += out_grad

class PAD(OP): 
  @staticmethod 
  def forward(x, args):
    assert isinstance(args, (tuple, list))
    out = np.pad(x.data, pad_width=args, mode='constant')
    return out

  def backward(self, out_grad, out): 
    w = [] 
    for i,j in zip(self.ctx, out.shape):
      w.append(slice(i[0], j-i[1], None))
    self.saved[0].grad += out_grad[tuple(w)]

class ROLL(OP): 
  @staticmethod
  def forward(x, shift, axis): 
    return np.roll(x.data, shift, axis)

  def backward(self, out_grad, out): 
    shift, axis = self.ctx
    self.saved[0].grad += np.roll(out_grad, -shift, axis)

# input: kernel and conv2d input shape, output toeplitz matrix
# fns: concatonate, slice, reshape, pad
# assert that the input shape is 4d
class SPARSE(OP): 
  @staticmethod
  def forward(x, *shape):
    _, _, kp, kq = x.shape
    _, _, xp, xq = shape
    pd_w = ((0,0),(0,0),(0,0),(0,xq-kq))
    pd = np.pad(x.data, pad_width = pd_w, mode = 'constant').reshape(-1,) # pad + reshape
    r = np.zeros(xq-kq+1)
    r[0] = x.data[0,0,0,0] # slice
    vr = np.concatenate((pd, r[1:])) #cat
    indx = np.arange(0, xq-kq+1) + np.expand_dims(np.arange(pd.shape[0]-1,-1,-1), 1) 
    indx = np.flip(indx, axis=[0]) # flip
    out = vr[indx] # idx
    return out

  def backward(x, *shape):
    pass
    # TODO: implement IDX(CAT(SLICE, RESHAPE(PAD)))









