import onnx
from class_GPU import GPU


class Node:
    def __init__(self, index, name, op_type, layer=0):
        self.index = index               # Numeric index (for internal reference)
        self.name = name                 # Original ONNX node name
        self.op_type = op_type           # Operation type (e.g., MatMul, Add)
        self.parents = []                # List of parent nodes
        self.gpu = None
        self.collective = False          # define operator as collective
        self.data_size = 0
        self.layer = layer
        self.id_d = 0
        self.id_t = 0
        self.id_p = 0

    def __repr__(self):
        return f"{self.name} ({self.op_type})"

