import onnx


# Custom Node class
class Node:

    def __init__(self, name, op_type):
        self.name = name            # node name (index)
        self.op_type = op_type      # operation type (e.g., MatMul, Add)
        self.parents = []           # list of parent nodes
   #     self.children = []          # list of child nodes

    def __repr__(self):    # print the object
        return f"{self.name} ({self.op_type})"

