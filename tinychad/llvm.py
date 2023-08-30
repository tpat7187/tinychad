import numpy as np 
import llvmlite.ir as ir
import llvmlite.binding as llvm
from typing import Union, Tuple, List, Optional
import tinychad.ops as ops
from ctypes import c_void_p, CFUNCTYPE
import ctypes

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
    self._bufs_ptr = [j.ctypes.data_as(c_void_p) for j in self._bufs]
    self.generated_fxns = set()
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
      ops.RELU: lambda builder, x: builder.select(builder.fcmp_unordered(">", x, ir.Constant(ir.FloatType(), 0)), x, ir.Constant(ir.FloatType(), 0)),

      ops.SUM: lambda builder, x, y: builder.fadd(x,y),
      ops.MAX: lambda builder, x, y: builder.select(builder.fcmp_unordered(">", x, y), x, y)
    }
      

  def parse_cache(self): 

    BinaryOPS = [ops.ADD, ops.SUB, ops.MUL, ops.DIV, ops.MATMUL]
    UnaryOPS = [ops.RELU, ops.LOG, ops.EXP, ops.NEG, ops.SQRT]
    ShapeOPS = [ops.SUM, ops.MAX]
    ReshapeOPS = [ops.RESHAPE, ops.SLICE, ops.TRANSPOSE, ops.PAD, ops.CAST]

    s, tt = [j[0] for j in self._cache], {}
    for j in range(len(self._cache)):
      tt[s[j]] = self.args[j]
    for j in self._cache: 
      if j[2] != ops.LOAD:
        input_args = [tt[j[4]], tt[j[5]]] if len(j) == 6 else [tt[j[4]]]
        output_arg = tt[j[0]]
        if j[2] in (BinaryOPS + UnaryOPS):
          self.elementwise_op(j[2], j[1], output_arg, input_args)
        if j[2] in ShapeOPS:
          args = j[3].op.ctx
          # pass in entire output buffer
          self.shape_op(j[2], j[3], output_arg, input_args, args)

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
  # there has to be a way to do some fusing for the codegen
  def elementwise_op(self, op, shapes, output_arg, input_args):
    # generate new function
    num_in_buffers = len(input_args)
    fxn_type = ir.FunctionType(void_t, [arr_t for _ in range(1+len(input_args))])
    fxn = ir.Function(self.mod, fxn_type, name = f"{str(op.__name__)}_{shapes[0]}")
    inp_block, loop_block, out_block = fxn.append_basic_block(name = 'entry'), fxn.append_basic_block(name = 'loop'), fxn.append_basic_block(name = 'out')
    inp_builder, loop_builder, out_builder = ir.IRBuilder(inp_block), ir.IRBuilder(loop_block), ir.IRBuilder(out_block)
    inp_builder.branch(loop_block)
    out_builder.ret_void()
    s_ptr, e_ptr = ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), np.prod(shapes))
    idx = loop_builder.phi(ir.IntType(32))
    idx.add_incoming(s_ptr, inp_block)
    av = loop_builder.load(loop_builder.gep(fxn.args[0], [idx]))
    inputs = [av]  
    if num_in_buffers > 1: inputs.append(loop_builder.load(loop_builder.gep(fxn.args[1], [idx])))

    out_ptr = loop_builder.gep(fxn.args[num_in_buffers], [idx])
    loop_builder.store(self.op_map[op](loop_builder, *tuple(inputs)), out_ptr)
    idx_n = loop_builder.add(idx, ir.Constant(ir.IntType(32), 1))
    idx.add_incoming(idx_n, loop_block)
    loop_builder.cbranch(loop_builder.icmp_unsigned("<", idx, e_ptr), loop_block, out_block)
    self.generated_fxns.add(fxn.name)
    self.main_builder.call(fxn, (*input_args, output_arg))

  # sum/max, axis is a accumulate with a stride
  # [1,2,3,4,5,6] (2,3) axis = 0 -> [1+4], [2+5], [3+6], 3 blocks 2 elements
  # [1,2,3,4,5,6] (2,3) axis = 1 -> [1+2+3], [4+5+6], 2 blocks 3 elements
  # if the stride changes between blocks we need to introduce another loop
  # EG:  [0,1,2,3,4,5,6,7] axis=0 -> [0,4]->(1)->[1,5]->(1)->[2,6]->(1)->[3,7] stride=4, block stride=1
  # EG2: [0,1,2,3,4,5,6,7] axis=1 -> [0,2]->(1)->[1,3]->(3)->[4,6]->(1)->[5,7] stride=2, block stride=(1,3)
  # EG3: [0,1,2,3,4,5,6,7] axis=2 -> [0,1]->(2)->[2,3]->(2)->[4,5]->(2)->[6,7] stride=1, block stride=2
  # loop 1: idx0, idx0*3 
  # loop 2: idx0, idx0+stride

  # if stride AND block_stride > 1 NEED new loop
  @staticmethod
  def get_strides(shape, axis):
    stride = 1
    for i in range(axis+1, len(shape)):
        stride *= shape[i]
    if axis == 0: block_stride = 1
    elif axis == len(shape) - 1: block_stride = stride * shape[axis]
    else: block_stride = stride * shape[axis]
    return stride, block_stride


  def shape_op(self, op, shapes, output_arg, input_args, args):
    # args[1] is keepdim, this doesn't really matter as far as memory patterns are concerned
    in_shape = shapes.op.saved[0].shape
    out_shape = shapes.shape

    shapes = list(in_shape)[::-1]
    _blocks = np.sum(out_shape)
    if args[0] != None:
      stride = [1]
      for s in reversed(shapes[:-1]): 
        stride.insert(0, stride[0]*s)
      stride = stride[args[0]]
    else: 
      stride=1

    fxn_type = ir.FunctionType(void_t, [arr_t, arr_t])
    fxn = ir.Function(self.mod, fxn_type, name = f"{str(op.__name__)}_{shapes[0]}")
    inp_block, loop_block, out_block = fxn.append_basic_block(name = 'entry'), fxn.append_basic_block(name = 'loop'), fxn.append_basic_block(name = 'out')
    inp_builder, loop_builder, out_builder = ir.IRBuilder(inp_block), ir.IRBuilder(loop_block), ir.IRBuilder(out_block)
    inp_builder.branch(loop_block)
    out_builder.ret_void()
    s_ptr, e_ptr = ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), _blocks)

    idx = loop_builder.phi(ir.IntType(32))
    idx.add_incoming(s_ptr, inp_block)
    indx, cache = idx, []
    print(_blocks, stride)
    for x in range(np.prod(shapes) // _blocks):
      if stride == 1:
        indx = loop_builder.mul(idx, ir.Constant(ir.IntType(32), np.prod(shapes)//_blocks))
        indx = loop_builder.add(indx, ir.Constant(ir.IntType(32), x))
        av = loop_builder.load(loop_builder.gep(fxn.args[0], [indx]))
      else: 
        av = loop_builder.load(loop_builder.gep(fxn.args[0], [indx]))
        indx = loop_builder.add(indx, ir.Constant(ir.IntType(32), stride))
      cache.append(av)

    out = cache[0]
    out = loop_builder.fadd(out, ir.Constant(ir.FloatType(), 0))
    for i in range(1, len(cache)):
      out = self.op_map[op](loop_builder, cache[i], out)

    out_ptr = loop_builder.gep(fxn.args[1], [idx]) 
    loop_builder.store(out, out_ptr)

    idx_n = loop_builder.add(idx, ir.Constant(ir.IntType(32), 1))
    idx.add_incoming(idx_n, loop_block)
    loop_builder.cbranch(loop_builder.icmp_unsigned("<", idx, e_ptr), loop_block, out_block)
    self.main_builder.call(fxn, (*input_args, output_arg))

  def _transpose(self, args):
    pass

  def _slice(self, args):
    pass

  def _cast(self, args):
    pass

  def _pad(self, args):
    pass

  # this should just be a simple copy, the content of the array doesnt change
  def _reshape(self, args):
    pass
  


