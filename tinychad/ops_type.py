from enum import Enum, auto

class UnaryOPS(Enum): RELU = auto(); NEG = auto(); LOG = auto(); EXP = auto(); SQRT = auto();
class BinaryOPS(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MATMUL = auto();
class ShapeOPS(Enum): MAX = auto(); SUM = auto();
class ReshapeOPS(Enum): RESHAPE = auto(); SLICE = auto(); PAD = auto(); TRANSPOSE = auto(); CAST = auto();
class LoadOPS(Enum): LOAD = auto();
