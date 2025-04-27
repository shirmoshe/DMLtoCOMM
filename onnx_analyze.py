import onnx
from class_Node import Node
from graphviz import Digraph
import webbrowser
from graphviz import Digraph
import webbrowser
from collections import defaultdict
import data_parllel
import re


def load_model(model_path):
    """Loads an ONNX model, performs shape inference, and prints input/output shapes for each layer."""
    model = onnx.load(model_path)
    model = onnx.shape_inference.infer_shapes(model)
#    onnx.checker.check_model(model)
    print("ONNX model is valid!\n")
    return model


def create_nodes(model):
    """Create Node objects and parent-child relationships based on ONNX model."""
    output_to_node = {}
    node_list = []

    # Create a Node object for each onnx_node
    for idx, onnx_node in enumerate(model.graph.node):
        # Use the original ONNX node name if it exists; otherwise generate one
        node_name = onnx_node.name if onnx_node.name else f"Node_{idx}"
        node = Node(index=idx, name=node_name, op_type=onnx_node.op_type)

        # assign layer
        match = re.search(r"/layers\.(\d+)/", node.name)
        if match:
            node.layer = int(match.group(1))
        else:
            node.layer = -1

        node_list.append(node)

        # Map each output tensor name to the Node that produced it
        for out in onnx_node.output:
            output_to_node[out] = node

    # Build parent-child links
    for idx, onnx_node in enumerate(model.graph.node):
        current_node = node_list[idx]
        for inp in onnx_node.input:
            if inp in output_to_node:
                parent_node = output_to_node[inp]
                current_node.parents.append(parent_node)
            else:
                # This is likely a model input, so we skip or handle differently
                pass

    return node_list


def group_layer(nodes):
    # create dict{key->layer, val->[]list of operators}
    layers = defaultdict(list)
    for node in nodes:
        layer_num = node.layer
        layers[layer_num].append(node)

    return dict(layers)


