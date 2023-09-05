from tinychad.tensor import tensor, Linear, Conv2d, BatchNorm2d
import numpy as np

# standard Conv->BatchNorm->Activation
class ConvBlock: 
  def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
    self.c1 = Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
    self.b1 = BatchNorm2d(out_channels)

  def __call__(self, x):
    x = self.c1(x)
    x = self.b1(x).relu()
    return x

  def forward(self, x): 
    x = self.c1(x)
    x = self.b1(x).relu()
    return x

class ResidualConvBlock: 
  def __init__(self, in_channels, out_channels, kernel_size, shortcut=None): 
    self.block = ConvBlock(in_channels, out_channels, kernel_size)
    self.shortcut = shortcut

  def __call__(self, x):
    residual = x
    residual = self.shortcut(residual)
    x = self.block(x)
    x = x + residual
    return x

class BottleNeck: 
  def __init__(self, in_channels, out_channels, r=4): 
    self.shortcut = ConvBlock(in_channels, out_channels, kernel_size=1)
    reduced_shape = out_channels // r
    self.c1 = ConvBlock(in_channels, reduced_shape, kernel_size=1)
    self.c2 = ConvBlock(reduced_shape, reduced_shape, kernel_size=3)
    self.c3 = ConvBlock(reduced_shape, out_channels, kernel_size=1)

  def __call__(self, x): 
    residual = x
    residual = self.shortcut(residual)
    x = self.c1(x)
    x = self.c2(x)
    x = self.c3(x)
    x = x + residual
    return x

class InvertedResidualBlock: 
  def __init__(self, in_channels, out_channels, r=4):
    self.shortcut = ConvBlock(in_channels, out_channels, kernel_size=1)
    expanded_shape = out_channels * r
    self.c1 = ConvBlock(in_channels, expanded_shape, kernel_size=1)
    self.c2 = ConvBlock(expanded_shape, expanded_shape, kernel_size=3)
    self.c3 = ConvBlock(expanded_shape, out_channels, kernel_size=1)

  def __call__(self, x): 
    residual = x
    residual = self.shortcut(residual)
    x = self.c1(x)
    x = self.c2(x)
    x = self.c3(x)
    x = x + residual
    return x

if __name__ == "__main__": 
  inp = tensor.randn(1,32,58,58)

  Conv1x1 = ConvBlock(32, 64, kernel_size=1)
  Conv3x3 = ConvBlock(32, 64, kernel_size=3)
  Conv1x1(inp)
  Conv3x3(inp)

  ResConvBlock1x1 = ResidualConvBlock(32,64,3, shortcut = ConvBlock(32,64,3))
  ResConvBlock1x1(inp)

  Bnec = BottleNeck(32,64)
  Bnec(inp)

  InvRes = InvertedResidualBlock(32,64)
  InvRes(inp)




