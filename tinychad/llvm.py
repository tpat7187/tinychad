import numpy as np 
import llvmlite.ir as ir
import llvmlite.binding as llvm
from typing import Union, Tuple, List, Optional
import tinychad.ops as ops
import ctypes

# this is just a lazy buffer in desguise
void_t = ir.VoidType()
arr_t = ir.PointerType(ir.FloatType())

class LLVMCodegen: 
  def __init__(self, _cache): 
    # LOAD actual data from buffers, initialize all other buffers to zeros
    _bufs = [j[3].data for j in _cache]
    self._cache, self.mod, self._bufs = _cache, ir.Module(), _bufs
    self.fun_t = ir.FunctionType(void_t, [arr_t for _ in range(len(self._bufs))])
    self.main = ir.Function(self.mod, self.fun_t, name = 'main')

    self.main_block  = self.main.append_basic_block(name = 'main')
    self.main_builder = ir.IRBuilder(self.main_block)

    self.out_block  = self.main.append_basic_block(name = 'exit')
    self.out_builder = ir.IRBuilder(self.out_block)
    self.out_builder.ret_void()

    self.args = self.main.args
    self.op_map = {ops.ADD : ir.IRBuilder.fadd, ops.SUB : ir.IRBuilder.fsub, ops.MUL : ir.IRBuilder.fmul}

    self._bufs_ptr = [j.ctypes.data_as(ctypes.c_void_p) for j in self._bufs]

  def parse_cache(self): 
    s, tt = [j[0] for j in self._cache], {}
    for j in range(len(self._cache)):
      tt[s[j]] = self.args[j]
    for j in self._cache: 
      if j[2] != ops.LOAD:
        # this is temporary will only work for binaryops
        output_arg = tt[j[0]]
        input_args = [tt[j[4]], tt[j[5]]]
        self.elementwise_op(j[2], j[1], output_arg, input_args)

  # unary + binary sans MATMUL
  def elementwise_op(self, op, shapes, output_arg, input_args):
    # generate new function
    addi_type = ir.FunctionType(void_t, [arr_t, arr_t, arr_t])
    addi = ir.Function(self.mod, addi_type, name = f"{str(op.__name__)}_{shapes[0]}")
    llvm_op = self.op_map[op]
    inp_block = addi.append_basic_block(name = 'entry')
    loop_block = addi.append_basic_block(name = 'loop')
    out_block = addi.append_basic_block(name = 'out')
    inp_builder, loop_builder, out_builder = ir.IRBuilder(inp_block), ir.IRBuilder(loop_block), ir.IRBuilder(out_block)
    inp_builder.branch(loop_block)
    out_builder.ret_void()
    s_ptr, e_ptr = ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), np.prod(shapes))
    idx = loop_builder.phi(ir.IntType(32))
    idx.add_incoming(s_ptr, inp_block)
    a, b, c = addi.args
    av = loop_builder.load(loop_builder.gep(a, [idx]))
    bv = loop_builder.load(loop_builder.gep(b, [idx]))
    c_ptr = loop_builder.gep(c, [idx])
    loop_builder.store(llvm_op(loop_builder, av, bv), c_ptr)
    idx_n = loop_builder.add(idx, ir.Constant(ir.IntType(32), 1))
    idx.add_incoming(idx_n, loop_block)
    loop_builder.cbranch(loop_builder.icmp_unsigned("<", idx, e_ptr), loop_block, out_block)
    self.main_builder.call(addi, (*input_args, output_arg))

  # sum/max
  def shape_op(self, op, axis, keedim):
    pass

  # slice, pad, transpose, reshape, cast
  def _reshape(self, args):
    pass
  


