import numpy as np 
import llvmlite.ir as ir
import llvmlite.binding as llvm
import ctypes

# input1, input2, output


class LLVMcodegen:
  def __init__(self): 
    self.mod = ir.Module() 

def _llvmBinaryOP(x, y, r, mod):

  # types
  void_t = ir.VoidType()
  arr_t = ir.PointerType(ir.DoubleType())
  fun = ir.FunctionType(void_t, [arr_t, arr_t, arr_t])

  module = mod
  func = ir.Function(module, fun, name = 'adda')

  #blocks
  inp_block = func.append_basic_block(name = 'entry')
  loop_block = func.append_basic_block(name = 'loop')
  out_block = func.append_basic_block(name = 'exit')
  
  inp_builder = ir.IRBuilder(inp_block)
  loop_builder = ir.IRBuilder(loop_block)
  out_builder = ir.IRBuilder(out_block)

  out_builder.ret_void()

  inp_builder.branch(loop_block)

  s_ptr = ir.Constant(ir.IntType(32), 0)
  e_ptr = ir.Constant(ir.IntType(32), len(r))

  idx = loop_builder.phi(ir.IntType(32))
  idx.add_incoming(s_ptr, inp_block)

  a, b, c = func.args

  # get value at index from input
  av = loop_builder.load(loop_builder.gep(a, [idx]))
  bv = loop_builder.load(loop_builder.gep(b, [idx]))
  c_ptr = loop_builder.gep(c, [idx]) 

  # store value from addition at index of r
  loop_builder.store(loop_builder.fadd(av, bv), c_ptr)

  # incremenet loop
  idx_n = loop_builder.add(idx, ir.Constant(ir.IntType(32), 1)) 
  idx.add_incoming(idx_n, loop_block)

  # if idx > size then stop
  loop_builder.cbranch(loop_builder.icmp_unsigned("<", idx, e_ptr), loop_block, out_block)
  return module


if __name__ == "__main__":
  x = np.random.randn(5,5)
  y = np.random.randn(5,5)
  r = np.zeros((5,5))

  llvmprg = LLVMcodegen()

  mod = _llvmBinaryOP(x, y, r, llvmprg.mod)

  print(mod)
  
