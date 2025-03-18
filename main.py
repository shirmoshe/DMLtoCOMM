import onnx
import netron
import onnx_analyze
import class_Node
import torch
from onnx2pytorch import ConvertModel

model_path = "C:/Users/shirm/PycharmProjects/Project/imagenet/resnet50-v1-7.onnx"  # Replace with model path

# Load the ONNX model
onnx_model = onnx_analyze.load_model(model_path)

# Visualize the ONNX model
#onnx_analyze.launch_netron(model_path)

# Create parent-child relationships
nodes_list = onnx_analyze.create_nodes(onnx_model)

# Print parent-child relationships
onnx_analyze.print_nodes(nodes_list)

# Create and print set of operators
ops = onnx_analyze.operators_list(onnx_model)



# Plot the directed graph with our nodes
# onnx_analyze.plot_onnx_graph(nodes_list)
















# Convert ONNX to PyTorch
# pytorch_model = ConvertModel(onnx_model)

# Show the converted PyTorch model
# print("Converted PyTorch Model:")
# print(pytorch_model)