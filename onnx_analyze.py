import onnx
import netron
import class_Node
import networkx as nx
import matplotlib.pyplot as plt
from class_Node import Node


import onnx

def load_model(model_path):
    """Loads an ONNX model, performs shape inference, and prints input/output shapes for each layer."""
    model = onnx.load(model_path)
    model = onnx.shape_inference.infer_shapes(model)  # Perform shape inference
    onnx.checker.check_model(model)  # Check if model is valid
    print("ONNX model is valid!\n")

    # Step 1: Extract tensor shapes from model inputs, value info, and outputs
    tensor_shapes = {}  # Store {tensor_name: shape}

    for tensor in model.graph.input:  # Model Inputs
        shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" for dim in tensor.type.tensor_type.shape.dim]
        tensor_shapes[tensor.name] = shape

    for tensor in model.graph.value_info:  # Intermediate tensors (after shape inference)
        shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" for dim in tensor.type.tensor_type.shape.dim]
        tensor_shapes[tensor.name] = shape

    for tensor in model.graph.output:  # Model Outputs
        shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" for dim in tensor.type.tensor_type.shape.dim]
        tensor_shapes[tensor.name] = shape

    # Step 2: Store only real operator inputs (not initializers)
    output_to_node = {}  # key = output_name, value = node.op_type
    for node in model.graph.node:
        for output in node.output:
            output_to_node[output] = node.op_type  # Store which node produced this output

    # Step 3: Print ONNX model layers with input/output shapes
    print("\nModel Layers:")
    for node in model.graph.node:
       # print(f"Raw ONNX Node Outputs: {node.output}")  # Check all outputs
       # print(f"Raw ONNX Node Inputs: {node.input}")  # Check all outputs

        # Keep only inputs that are outputs of another node (ignore weights & biases)
        real_inputs = [inp for inp in node.input if inp in output_to_node]

        # Get input & output shapes
        input_shapes = {inp: tensor_shapes.get(inp, "Shape Not Found") for inp in real_inputs}
        output_shapes = {out: tensor_shapes.get(out, "Shape Not Found") for out in node.output}

        print(f"  - Operation: {node.op_type}")
        print(f"    Inputs: {input_shapes}")  # Shows input names & their shapes
        print(f"    Outputs: {output_shapes}")  # Shows output names & their shapes
        print("-" * 40)



    return model


def launch_netron(model_path):
    """Launches Netron to visualize the ONNX model in a web browser"""
    netron.start(model_path)


def create_nodes(model):
    """Get onnx model and create parent-child relationships"""
    # Build a mapping of outputs to their producing nodes
    output_to_node = {}      # key= output, val= Node
    node_list = []

    for idx, onnx_node in enumerate(model.graph.node):
        node = Node(name=idx, op_type=onnx_node.op_type)
        node_list.append(node)

        for out in onnx_node.output:
            output_to_node[out] = node

    # keep only inputs from another node
    output_to_node = {}  # key = output_name, value = node.op_type
    for node in model.graph.node:
        for output in node.output:
            output_to_node[output] = node.op_type  # Store which node produced this output

    for idx, onnx_node in enumerate(model.graph.node):
        current_node = node_list[idx]
        real_inputs = [inp for inp in onnx_node.input if inp in output_to_node]  # Keep only inputs from another node
        for inp in real_inputs:
            parent_node = output_to_node.get(inp)   # If no parent, assume it's a model input
            current_node.parents.append(parent_node)

    return node_list

def print_nodes(nodes_list):
    """Print parent-child relationships"""

    print("**Parent-Child Relationships:**\n")
    for node in nodes_list:
        if node.parents:
            print(f"{node.parents} -> {node.name} ({node.op_type})")

            '''
             for parent in node.parents:
                if parent == "Input":
                    print(f"Input ->{node.name} ({node.op_type})")
                else:
                    print(f"{node.name} ({node.op_type}) ← {parent.name} ({parent.op_type})")
            '''

        else:
            print(f"No parent -> {node.name} ({node.op_type})")

def operators_list(model):
    """Prints a list of operators in the ONNX model"""
    ops = set()  # set a list with no duplicate operators

    for node in model.graph.node:
        ops.add(node.op_type)  # add operator type to the set

    print("Operators in the Model:")
    for op in sorted(ops):  # sort for easier readability
        print(f"  - {op}")
    return ops


def plot_onnx_graph(nodes_list):
    """Plots the model as a directed graph using parent-child relationships"""
    G = nx.DiGraph()

    # Add nodes and store labels
    node_labels = {}  # Dictionary for storing labels separately

    for node in nodes_list:
        G.add_node(node.name)  # Add node to graph
        node_labels[node.name] = node.op_type  # Store label separately

        for parent in node.parents:
            if parent != "Input":  # Ensure "Input" nodes don't cause errors
                G.add_edge(parent.name, node.name)  # Add edge from parent to child
            else:
                G.add_edge("Input", node.name)  # Input to first nodes

    # Define layout
    pos = nx.spring_layout(G, seed=42)  # Positioning of nodes

    # Draw the graph
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=3000,
            node_color="skyblue", edge_color="black", font_size=8,
            font_weight="bold", arrows=True)

    plt.title("ONNX Model Graph (Parent-Child Relationship)")
    plt.show()