import numpy as np 

#### OPS CLASS ####

class OP: 
  def __init__(self, saved = None):
    self.arg = type(self).__name__
    self.saved = np.array(saved)

class ADD(OP): 
  def __init__(self, saved = None): 
    self.arg = type(self).__name__
    self.saved = np.array(saved)

  def forward(x, y): 
    return tensor(x.data + y.data, op = ADD(saved = (x,y)))
  
  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad
    self.saved[1].grad += out_grad

class MATMUL(OP): 
  def __init__(self, saved = None): 
    self.arg = type(self).__name__
    self.saved = np.array(saved)

  def forward(x, y): 
    return tensor(np.dot(x.data, y.data), op = MATMUL(saved = (x,y)))

  def backward(self, out_grad, out):
    self.saved[0].grad += np.dot(out_grad, self.saved[1].T().data)
    self.saved[1].grad += np.dot(self.saved[0].T().data, out_grad)

class LOAD(OP): 
  def __init__(self, saved = None):
    self.arg = type(self).__name__
    self.saved = saved

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
  def __matmul__(self,x): return self.dot(x)

  # fundamental OPS
  def add(self, x): return ADD.forward(self, x)
  def dot(self, x): return MATMUL.forward(self, x)

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
    print(*reversed(topo), sep="\n") if track == True else None
    return topo

  def backward(self, track = False): 
    self.grad = np.ones(self.grad.shape)
    for x in reversed(self.toposort(track)): 
      x.op.backward(x.grad, x.data)

def test(): 
  x = tensor.randn(1,784)
  y = tensor.randn(784,128)
  z = tensor.randn(128,10)

  w = x @ y
  j = w @ z

  j.backward(track = True)

if __name__ == "__main__":
  test()


