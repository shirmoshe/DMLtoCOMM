import onnx

# Custom Node class
class Node:
    def __init__(self, index, name, op_type):
        self.index = index               # Numeric index (for internal reference)
        self.name = name                 # Original ONNX node name
        self.op_type = op_type           # Operation type (e.g., MatMul, Add)
        self.parents = []                # List of parent nodes
        self.gpu_naum = 0
        self.collective = False           # define operator as collective

    def __repr__(self):
        return f"{self.name} ({self.op_type})"

