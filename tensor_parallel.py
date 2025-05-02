from class_GPU import GPU
from class_Node import Node

# operator list for tensor parallel
TENSOR_PARALLEL_OPS = {
    "MatMul", "MatMulNBits", "Gemm", "Conv", "ConvTranspose",
    "Linear", "FullyConnected", "BatchMatMul"
}


def is_tensor_parallel_candidate(node):
    return node.op_type in TENSOR_PARALLEL_OPS and any("weight" in inp for inp in node.inputs)




