import numpy as np 
import llvmlite.ir as ir
import llvmlite.binding as llvm
import ctypes

void_t = ir.VoidType()
arr_t= ir.PointerType(ir.DoubleType())

class LLVMcodegen:
  def __init__(self, _bufs):
    self.mod, self._bufs = ir.Module(), _bufs
    self.fun_t = ir.FunctionType(void_t, [arr_t for _ in range(len(self._bufs))])
    self.main = ir.Function(self.mod, self.fun_t, name = 'main')
    
    self.main_block  = self.main.append_basic_block(name = 'main')
    self.main_builder = ir.IRBuilder(self.main_block)

    self.out_block  = self.main.append_basic_block(name = 'exit')
    self.out_builder = ir.IRBuilder(self.out_block)
    self.out_builder.ret_void()

  def _compile(self): 
    # end function calls / instructions
    self.main_builder.branch(self.out_block)

    input_ir = self.mod
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    print(self.mod)

    llvm_ir = str(input_ir)


    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly("") 
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()

    func_ptr = engine.get_function_address("main")
    llvmfunc = ctypes.CFUNCTYPE(None, *[ctypes.c_void_p for _ in self._bufs])(func_ptr)

    return llvmfunc

# should these be class methods?
def _add(llvmcg, x, y, r):
  addi_type = ir.FunctionType(void_t, [arr_t, arr_t, arr_t]) 
  addi = ir.Function(llvmcg.mod, addi_type, name = 'addi')

  _binaryOP(llvmcg.mod, x, y, r, ir.IRBuilder.fadd, addi)

  x, y, r = addi.args

  llvmcg.main_builder.call(addi, (x,y,r))

def _sub(mod, x, y, r):
  subi_type = ir.FunctionType(void_t, [arr_t, arr_t, arr_t]) 
  subi = ir.Function(mod, subi_type, name = 'subi')


  _binaryOP(mod, x, y, r, ir.IRBuilder.fsub, subi)

def _mul(mod, x, y, r):
  muli_type = ir.FunctionType(void_t, [arr_t, arr_t, arr_t]) 
  muli = ir.Function(mod, muli_type, name = 'muli')

  _binaryOP(mod, x, y, r, ir.IRBuilder.fmul, muli)

# get function name to work
def _binaryOP(mod, x, y, r, op, fxn): 
  inp_block = fxn.append_basic_block(name = 'entry')
  loop_block = fxn.append_basic_block(name = 'loop')
  out_block = fxn.append_basic_block(name = 'out')

  inp_builder = ir.IRBuilder(inp_block)
  loop_builder = ir.IRBuilder(loop_block)
  out_builder = ir.IRBuilder(out_block)

  inp_builder.branch(loop_block)

  s_ptr = ir.Constant(ir.IntType(32), 0)
  e_ptr = ir.Constant(ir.IntType(32), 3)
  idx = loop_builder.phi(ir.IntType(32))
  idx.add_incoming(s_ptr, inp_block)

  a, b, c = fxn.args

  av = loop_builder.load(loop_builder.gep(a, [idx]))
  bv = loop_builder.load(loop_builder.gep(b, [idx]))
  c_ptr = loop_builder.gep(c, [idx])

  loop_builder.store(op(loop_builder, av, bv), c_ptr)

  idx_n = loop_builder.add(idx, ir.Constant(ir.IntType(32), 1))
  idx.add_incoming(idx_n, loop_block)

  loop_builder.cbranch(loop_builder.icmp_unsigned("<", idx, e_ptr), loop_block, out_block)

  out_builder.ret_void()
  return mod, fxn

if __name__ == "__main__":
  x = np.ones((3))
  y = np.ones((3))
  r = np.zeros((3))

  llvm_prg = LLVMcodegen([x,y,r])

  _add(llvm_prg, x, y, r)

  # convert to pointers
  xp = x.ctypes.data_as(ctypes.c_void_p)
  yp = y.ctypes.data_as(ctypes.c_void_p)
  rp = r.ctypes.data_as(ctypes.c_void_p)

  llvmfunc = llvm_prg._compile()

  res = llvmfunc(xp, yp, rp)







  
