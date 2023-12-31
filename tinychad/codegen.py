from __future__ import annotations
import ctypes, subprocess, tempfile
from typing import Union, Tuple, Optional, List, Dict
from tinychad.ops_type import UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS, LoadOPS, TokenType
from tinychad.helpers import DEBUG
import numpy as np 

class ExecuteCProgram:
  def __init__(self, prg:str, bufs, fxn_name: str): 
    self.prg = prg
    self.args = [np.ctypeslib.as_ctypes(child.data) for child in bufs.children] + [np.ctypeslib.as_ctypes(bufs.data)]

    self.fxn_name = fxn_name
    self.dll = self.compile()

  def compile(self) -> ctypes.CDLL:
    with tempfile.NamedTemporaryFile(suffix='.so', delete=False) as fp:
      subprocess.check_output(
        ['clang', '-shared', '-march=native', '-O3', '-Wall', '-Werror',
          '-x', 'c', '-fPIC', '-o', fp.name, '-'],
        input=self.prg.encode('utf-8')
      )
      lib = ctypes.CDLL(fp.name)
      return lib

  def run(self) -> None: 
    cfun = getattr(self.dll, self.fxn_name)
    cfun(*self.args)


class C_Codegen:  
  def __init__(self, tokens): 
    self.tokens = tokens
    self.kernel = self.generate_kernel(tokens)

  def generate_kernel(self, toks):
    lines = [] 
    for tok in toks:
      if tok.type == TokenType.FUNCSTART: 
        cg = f"void {tok.args[0]}({', '.join(['float* ' + _ for _ in tok.args[1]])}) {{"
        tok.codegen = cg 
        lines.append(cg)

      elif tok.type == TokenType.FUNCEND: lines.append("}")

      elif tok.type == TokenType.LOOPSTOP: lines.append("}")

      elif tok.type == TokenType.LOOPSTART:
        cg = f"for (int {tok.reg}={tok.start}; {tok.reg}<{tok.iters}; {tok.reg}+={tok.inc}) {{"
        tok.codegen = cg
        lines.append(cg)

      # load from buffer
      elif tok.type == TokenType.LOAD: 
        cg = f"float {tok.reg} = {tok.args[0]}[{tok.args[1]}];"
        tok.codegen = cg 
        lines.append(cg)

      # stores into buffer
      elif tok.type == TokenType.GLOBAL: 
        cg = f"{tok.reg}[{tok.args[1][0].reg}] = {tok.args[0].reg};"
        tok.codegen = cg
        lines.append(cg)
      
      # stores into float
      elif tok.type == TokenType.LOCAL: 
        if tok.args.type == TokenType.OP: 
          op_string = self.codegen_operation(tok.args) 
          cg = f"float {tok.reg}={op_string};"
          tok.codegen = cg 
          lines.append(cg)

      elif tok.type == TokenType.DEFINE_ACC: 
        cg = f"float {tok.reg}=0;"
        tok.codegen = cg
        lines.append(cg)

      elif tok.type == TokenType.ACC: 
        print(tok.args[0])
        cg = f"{tok.args[1].reg} += {tok.args[0].reg};"
        lines.append(cg)

    kern = '\n'.join(lines)
    if DEBUG: print(kern)

    return kern

  def codegen_operation(self, token):
    if token.args[0] in BinaryOPS:
      op_token = ops_to_toks[token.args[0]]
      cg = f"{f' {op_token} '.join([f'''{x.args[0]}[{x.args[1]}]''' for x in token.args[1]])}"
      token.codegen = cg 
      return cg
    else: 
      op_token =  ops_to_toks[token.args[0]]
      outreg = token.args[1][0].reg
      cg = f"float {token.reg} = {op_token}({outreg});"
      token.codegen = cg 
      return cg

ops_to_toks = { 
  BinaryOPS.ADD: '+',
  BinaryOPS.SUB: '-',
  BinaryOPS.MUL: '*',
  BinaryOPS.DIV: '/',
  UnaryOPS.RELU: 'relu'
}
