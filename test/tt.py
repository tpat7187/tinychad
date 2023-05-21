from tinychad.tensor import tensor
import numpy as np



x = tensor.randn(5,1).data

print(x)

x = np.broadcast_to(x, (5,5))

print(x)



