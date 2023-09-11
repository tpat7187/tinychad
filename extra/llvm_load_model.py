from tinychad.tensor import tensor, BatchNorm2d, Linear, Conv2d
from tinychad.llvm import LLVMCodegen
import numpy as np
from ctypes import c_byte, CFUNCTYPE, c_void_p
import llvmlite.ir as ir
import llvmlite.binding as llvm

# instantiate a LLVM program

'''
x = tensor.randn(28,28).reshape(1,-1)

l1 = Linear(784, 128)
l2 = Linear(128,10)

out = l1(x).relu()
out2 = l2(out).logsoftmax(axis=1)

x = tensor.ones((5,5))
y = tensor.ones((5,5))

out2 = x.dot(y)

cache = out2.get_buffers()
codegen = LLVMCodegen(cache)
codegen.compile()
print(out2.detach())
'''


module = ir.Module(name="byte_array_module")

def create_buffer(module, byte_array, name):
    length = len(byte_array)
    byte_array_type = ir.ArrayType(ir.IntType(8), length)
    global_variable = ir.GlobalVariable(module, byte_array_type, name=f"global_byte_array{name}")
    global_variable.initializer = ir.Constant(byte_array_type, byte_array)
    global_variable.linkage = 'dso_local'
    global_variable.align = 16
    float_ptr_type = ir.PointerType(ir.FloatType())
    float_ptr = ir.GlobalVariable(module, float_ptr_type, name=f"buf_{name}")
    float_ptr.initializer = ir.Constant(float_ptr_type, global_variable.bitcast(float_ptr_type))
    float_ptr.linkage = 'dso_local'
    float_ptr.align = 8

    return float_ptr

def create_empty_buffer(module, size, name):
    length = np.prod(size)
    float_array_type = ir.ArrayType(ir.FloatType(), length)
    float_array = ir.GlobalVariable(module, float_array_type, name = f"buf_{name}")
    float_array.initializer = ir.Constant(float_array_type, None)
    float_array.linkage = 'dso_local'
    float_array.align = 16

    return float_array

x = tensor.ones((10,10))
y = tensor.ones((10,10))

void_t = ir.VoidType()
arr_t = ir.PointerType(ir.FloatType())
arr_ptr = ir.PointerType(ir.PointerType(ir.FloatType()))

byte_array = x.detach().tobytes()
byte_array2 = y.detach().tobytes()

x1= create_buffer(module, list(byte_array), 1)
x2= create_buffer(module, list(byte_array2), 2)

y =create_empty_buffer(module, 100, 3)

output = np.zeros((10,10)).astype(np.float32)

main_type= ir.FunctionType(void_t, [arr_t])
main = ir.Function(module, main_type, name = "Main")
output_ptr = main.args[0]
main_block = main.append_basic_block(name = 'main')
out_block = main.append_basic_block(name = 'out')

main_builder = ir.IRBuilder(main_block)
main_out_builder = ir.IRBuilder(out_block)

buf_1_ptr = main_builder.load(x1, name= 'buf1_ptr')
buf_2_ptr = main_builder.load(x2, name= 'buf2_ptr')
buf_3_ptr = main_builder.gep(y, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)], name= 'buf3_ptr')

addi_type = ir.FunctionType(void_t, [arr_t, arr_t, arr_t])
addi = ir.Function(module, addi_type, name = "test_add")
inp_block = addi.append_basic_block(name = 'entry')
loop_block = addi.append_basic_block(name = 'loop')
out_block = addi.append_basic_block(name = 'out')
inp_builder, loop_builder, out_builder = ir.IRBuilder(inp_block), ir.IRBuilder(loop_block), ir.IRBuilder(out_block)
inp_builder.branch(loop_block)
out_builder.ret_void()
s_ptr, e_ptr = ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), np.prod(output.shape))
idx = loop_builder.phi(ir.IntType(32))
idx.add_incoming(s_ptr, inp_block)
av = loop_builder.load(loop_builder.gep(addi.args[0], [idx]))
bv = loop_builder.load(loop_builder.gep(addi.args[1], [idx]))
c_ptr = loop_builder.gep(addi.args[2], [idx])
loop_builder.store(loop_builder.fadd(av, bv), c_ptr)
idx_n = loop_builder.add(idx, ir.Constant(ir.IntType(32), 1))
idx.add_incoming(idx_n, loop_block)
loop_builder.cbranch(loop_builder.icmp_unsigned("<", idx_n, e_ptr), loop_block, out_block)
main_builder.call(addi, [buf_1_ptr, buf_3_ptr, output_ptr])

main_builder.branch(out_block)
main_out_builder.ret_void()

print(module)

input_ir = module
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
main_ptr = engine.get_function_address("Main")
cfunc = CFUNCTYPE(None, c_void_p)(main_ptr)
cfunc(output.ctypes.data_as(c_void_p))


print(output)



