import numpy as np 

#### OPS ####

class OP: 
  def __init__(self, saved = None, ctx = None):
    self.arg = type(self).__name__
    self.saved = np.array(saved)
    self.ctx = ctx

class LOAD(OP): 
  def __init__(self, saved = None):
    self.arg = type(self).__name__
    self.saved = saved

# binary ops
class ADD(OP): 
  @staticmethod
  def forward(x, y): 
    return tensor(x.data + y.data, op = ADD(saved = [x,y]))
  
  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad
    self.saved[1].grad += out_grad

class SUB(OP): 
  @staticmethod
  def forward(x, y): 
    return tensor(x.data - y.data, op = SUB(saved = [x,y]))
  
  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad
    self.saved[1].grad += out_grad

class MATMUL(OP): 
  @staticmethod
  def forward(x, y): 
    return tensor(np.dot(x.data, y.data), op = MATMUL(saved = [x,y]))

  def backward(self, out_grad, out):
    self.saved[0].grad += np.dot(out_grad, self.saved[1].T().data)
    self.saved[1].grad += np.dot(self.saved[0].T().data, out_grad)

# unary ops
class SUM(OP):
  @staticmethod
  def forward(x, axis):
    return tensor(np.array([x.data.sum(axis=axis)]), op = SUM(saved = [x,], ctx = axis))

  # TODO: write broadcasting so that this will work when we write NLL
  '''
    basically numpy broadcasting rules and pytorch broadcasting rules are a little different
    we need to write our own broadcasting so that we can pass gradients back like: 
    [5x1 to 5x5] when we do ops dimension wise
 '''
  def backward(self, out_grad, out):
    self.saved[0].grad += out_grad 


#### TENSOR CLASS ####
class tensor: 
  def __init__(self, data, op = LOAD(), requires_grad = False):
    self.data = np.array(data, dtype = np.float32)
    self.op = op 
    self.grad = np.zeros(self.data.shape)

  def ones(*shape): return tensor(np.ones(*shape))
  def randn(*shape): return tensor(np.random.randn(*shape))
  def eye(shape): return tensor(np.eye(shape))
  def zeros(*shape): return tensor(np.zeros(*shape))

  def __repr__(self): 
    return f"op = <{self.op.arg}>: shape = {self.data.shape}: grad_shape = {self.grad.shape}"

  def __add__(self,x): return self.add(x)
  def __sub__(self,x): return self.sub(x)
  def __matmul__(self,x): return self.dot(x)

  def add(self, x): return ADD.forward(self, x)
  def sub(self, x): return SUB.forward(self, x)
  def dot(self, x): return MATMUL.forward(self, x)

  def sum(self, axis = None): return SUM.forward(self, axis)

  # unary ops
  def T(self): return tensor(self.data.transpose())

  # i think this works
  def toposort(self, track): 
    topo, vis = [], []
    def _toposort(s): 
      if s not in vis: 
        vis.append(s)
        #print(s) if track == True else None
        if not isinstance(s.op, LOAD):
          for child in s.op.saved: 
            _toposort(child)
          topo.append(s)
    _toposort(self)
    
    # should we include load ops
    return topo

  def backward(self, track = False): 
    self.grad = np.ones(self.grad.shape)
    for x in reversed(self.toposort(track)): 
      print(x, sep="\n") if track == True else None
      x.op.backward(x.grad, x.data)

def test(): 
  # mnist shapes
  inp = tensor.randn(10,784)

  l1_w = tensor.randn(784,128) 
  l1_b = tensor.randn(10,128)

  l2_w = tensor.randn(128,10)
  l2_b = tensor.randn(10,10)

  layer1 = inp @ l1_w + l1_b 
  layer2 = layer1 @ l2_w + l2_b
  layer3 = layer2.sum()

  layer3.backward(track = True)


if __name__ == "__main__":
  test()



  














