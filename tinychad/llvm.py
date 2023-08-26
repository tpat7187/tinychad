import numpy as np 
import llvmlite.ir as ir
import llvmlite.binding as llvm
from typing import Union, Tuple, List, Optional
import tinychad.ops as ops
from ctypes import c_void_p, CFUNCTYPE
import ctypes

# this is just a lazy buffer in desguise
void_t = ir.VoidType()
arr_t = ir.PointerType(ir.FloatType())

class LLVMCodegen: 
  def __init__(self, _cache): 
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
    # llvm.fma intrinsic can perform fused multiply add, this may be useful for writing matmul kernel 
    # supposedly there is an intrinsic function already for matmuls
    self.op_map = {
      ops.ADD : lambda builder, x, y: builder.fadd(x,y),
      ops.SUB : lambda builder, x, y: builder.fsub(x,y),
      ops.MUL : lambda builder, x, y: builder.fmul(x,y),
      ops.DIV : lambda builder, x, y: builder.fdiv(x,y),

      ops.NEG: lambda builder, x: builder.fmul(x, ir.Constant(ir.FloatType(), -1)),
      ops.EXP: lambda builder, x: builder.call(builder._block.module.declare_intrinsic('llvm.exp', [ir.FloatType()]), [x]),
      ops.LOG: lambda builder, x: builder.call(builder._block.module.declare_intrinsic('llvm.log', [ir.FloatType()]), [x]),
      ops.SQRT: lambda builder, x: builder.call(builder._block.module.declare_intrinsic('llvm.sqrt', [ir.FloatType()]), [x]),
      ops.RELU: lambda builder, x: builder.select(builder.fcmp_unordered(">", x, ir.Constant(ir.FloatType(), 0)), x, ir.Constant(ir.FloatType(), 0))
    }
      

    self._bufs_ptr = [j.ctypes.data_as(c_void_p) for j in self._bufs]

  def parse_cache(self): 
    s, tt = [j[0] for j in self._cache], {}
    for j in range(len(self._cache)):
      tt[s[j]] = self.args[j]
    for j in self._cache: 
      if j[2] != ops.LOAD:
        input_args = [tt[j[4]], tt[j[5]]] if len(j) == 6 else [tt[j[4]]]
        output_arg = tt[j[0]]
        self.elementwise_op(j[2], j[1], output_arg, input_args)

  def compile(self): 
    self.parse_cache()
    self.main_builder.branch(self.out_block)
    input_ir = self.mod
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm_ir = str(input_ir)

    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly("") 
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    mod = llvm.parse_assembly(llvm_ir)
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    main_ptr = engine.get_function_address("main")
    cfunc = CFUNCTYPE(None, *[c_void_p for j in self._cache])(main_ptr)
    cfunc(*self._bufs_ptr)

  # unary + binary sans MATMUL
  def elementwise_op(self, op, shapes, output_arg, input_args):
    # generate new function
    num_in_buffers = len(input_args)
    fxn_type = ir.FunctionType(void_t, [arr_t for _ in range(1+len(input_args))])
    fxn = ir.Function(self.mod, fxn_type, name = f"{str(op.__name__)}_{shapes[0]}")
    llvm_op = self.op_map[op]
    inp_block = fxn.append_basic_block(name = 'entry')
    loop_block = fxn.append_basic_block(name = 'loop')
    out_block = fxn.append_basic_block(name = 'out')
    inp_builder, loop_builder, out_builder = ir.IRBuilder(inp_block), ir.IRBuilder(loop_block), ir.IRBuilder(out_block)
    inp_builder.branch(loop_block)
    out_builder.ret_void()
    s_ptr, e_ptr = ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), np.prod(shapes))
    idx = loop_builder.phi(ir.IntType(32))
    idx.add_incoming(s_ptr, inp_block)
    av = loop_builder.load(loop_builder.gep(fxn.args[0], [idx]))
    if num_in_buffers > 1: 
      bv = loop_builder.load(loop_builder.gep(fxn.args[1], [idx])) 
      inputs = tuple((av, bv))
    else: 
      inputs = tuple((av,))


    out_ptr = loop_builder.gep(fxn.args[num_in_buffers], [idx])
    loop_builder.store(llvm_op(loop_builder, *inputs), out_ptr)
    idx_n = loop_builder.add(idx, ir.Constant(ir.IntType(32), 1))
    idx.add_incoming(idx_n, loop_block)
    loop_builder.cbranch(loop_builder.icmp_unsigned("<", idx, e_ptr), loop_block, out_block)
    self.main_builder.call(fxn, (*input_args, output_arg))

  # sum/max
  def shape_op(self, op, axis, keedim):
    pass

  # slice, pad, transpose, reshape, cast
  def _reshape(self, args):
    pass
  


