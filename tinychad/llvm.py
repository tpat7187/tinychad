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
        if j[2] in ReshapeOPS: 
          self._reshape(j[2], j[1], output_arg, input_args)

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
  # IF 0 < axis < len(in_shape) we are going to have a block_stride > 1

  # if stride AND block_stride > 1 NEED new loop
  # stride and block_stride function

  @staticmethod
  def get_strides(shape, axis):
    stride = 1
    if axis == None: return stride
    for i in range(axis+1, len(shape)):
        stride *= shape[i]
    return stride

  @staticmethod 
  def get_block_stride(shape, axis, stride):
    if axis and 0 < axis < len(shape)-1: return stride * shape[axis]
    else: return None


  # args[1] is keepdim, this doesn't really matter as far as memory patterns are concerned
  def shape_op(self, op, shapes, output_arg, input_args, args):
    in_shape, out_shape = shapes.op.saved[0].shape, shapes.shape
    dim, _blocks = len(in_shape), np.prod(in_shape) // np.sum(out_shape)
    stride = LLVMCodegen.get_strides(in_shape, args[0])
    block_stride = LLVMCodegen.get_block_stride(in_shape, args[0], stride)
    fxn_type = ir.FunctionType(void_t, [arr_t, arr_t])
    fxn = ir.Function(self.mod, fxn_type, name = f"{str(op.__name__)}_{in_shape[0]}_{args[0]}")
    #fxn.attributes._known = fxn.attributes._known.union(frozenset(['"no-nans-fp-math"="true"']))
    #fxn.attributes.add('"no-nans-fp-math"="true"')
    inp_block = fxn.append_basic_block(name = 'entry')
    if block_stride: 
      global_block = fxn.append_basic_block(name='globalidx')
      global_builder = ir.IRBuilder(global_block)
    local_block = fxn.append_basic_block(name='localidx')
    local_builder = ir.IRBuilder(local_block)
    inp_builder = ir.IRBuilder(inp_block)
    # TODO: setup global idx
    if block_stride:
      global_s, global_e = ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 2)
      global_block_exit = fxn.append_basic_block(name='globalidx_exit')
      global_builder_exit = ir.IRBuilder(global_block_exit)
      global_idx = global_builder.phi(ir.IntType(32))
      inp_builder.branch(global_block)
      global_builder.branch(local_block)
    else:
      inp_builder.branch(local_block)
    
    out_block = fxn.append_basic_block(name = 'out')
    out_builder = ir.IRBuilder(out_block)
    out_builder.ret_void()

    # local strides and shit
    print(stride, block_stride)
    local_s, local_e = ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), np.prod(out_shape))
    local_idx = local_builder.phi(ir.IntType(32))
    # define how the local_idx moves
    indx, cache = local_idx, []
    if block_stride:
      local_idx.add_incoming(local_s, global_block)
      global_idx.add_incoming(global_s, inp_block)
      block_stride_idx = local_builder.mul(global_idx, ir.Constant(ir.IntType(32), block_stride))
      indx = local_builder.add(local_idx, block_stride_idx)
    else:
      local_idx.add_incoming(local_s, inp_block)
    # BLOCK SIZE (how many adds need to be performed)
    block_size = in_shape[args[0]] if args[0] != None else np.prod(in_shape)
    for x in range(block_size):
      if stride == 1: 
        indx = local_builder.mul(local_idx, ir.Constant(ir.IntType(32), block_size))
        indx = local_builder.add(indx, ir.Constant(ir.IntType(32), x))
        av = local_builder.load(local_builder.gep(fxn.args[0], [indx], inbounds=True))
      else:
        av = local_builder.load(local_builder.gep(fxn.args[0], [indx], inbounds=True))
        indx = local_builder.add(indx, ir.Constant(ir.IntType(32), stride))
      cache.append(av)
    out = cache[0]
    out = local_builder.fadd(out, ir.Constant(ir.FloatType(), 0))
    for i in range(1, len(cache)):
      out = self.op_map[op](local_builder, cache[i], out)

    # need to store: global_idx * local*idx
    local_e_ptr = local_builder.gep(fxn.args[1], [local_idx], inbounds=True)
    local_builder.store(out, local_e_ptr)
    local_e_n = local_builder.add(local_idx, ir.Constant(ir.IntType(32), 1))
    local_idx.add_incoming(local_e_n, local_block)
    if block_stride:
      local_builder.cbranch(local_builder.icmp_unsigned("==", local_idx, local_e_n), global_block_exit, local_block)
      global_e_n = global_builder_exit.add(global_idx, ir.Constant(ir.IntType(32), 1))
      global_idx.add_incoming(global_e_n, global_block_exit)
      global_builder_exit.cbranch(global_builder_exit.icmp_unsigned("==", global_idx, global_e), out_block, global_block)
    else:
      local_builder.cbranch(local_builder.icmp_unsigned("==", local_e_n, local_e), out_block, local_block)
    self.main_builder.call(fxn, (*input_args, output_arg))





  def _transpose(self, args):
    pass

  # plan for this is that args contains the slices so we simply read from the slices as our gep 
  def _slice(self, args):
    pass

  def _cast(self, args):
    pass

  def _pad(self, args):
    pass

  # this should just be a simple elementwise copy
  def _reshape(self, op, shapes, output_arg, input_args):
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
    out_ptr = loop_builder.gep(output_arg, [idx])
    loop_builder.store(av, out_ptr)
    idx_n = loop_builder.add(idx, ir.Constant(ir.IntType(32), 1))
    idx.add_incoming(idx_n, loop_block)
    loop_builder.cbranch(loop_builder.icmp_unsigned("<", idx, e_ptr), loop_block, out_block)
    self.generated_fxns.add(fxn.name)
    self.main_builder.call(fxn, (*input_args, output_arg))
    pass
  


