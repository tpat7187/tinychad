import numpy as np 
import ops as ops


#### TENSOR CLASS ####
class tensor: 
  def __init__(self, data, op = ops.LOAD(), requires_grad = False):
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

  def add(self, x): return tensor(ops.ADD.forward(self, x), op = ops.ADD(saved = [self,x]))
  def sub(self, x): return tensor(ops.SUB.forward(self, x), op = ops.SUB(saved = [self,x]))
  def dot(self, x): return tensor(ops.MATMUL.forward(self, x), op = ops.MATMUL(saved = [self,x]))

  def sum(self, axis = None): return tensor(ops.SUM.forward(self, axis), op = ops.SUM(saved = [self,], ctx=axis))

  # unary ops
  def T(self): return tensor(self.data.transpose())

  # i think this works
  def toposort(self, track): 
    topo, vis = [], []
    def _toposort(s): 
      if s not in vis: 
        vis.append(s)
        #print(s) if track == True else None
        if not isinstance(s.op, ops.LOAD):
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
















