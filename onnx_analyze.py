import onnx
from class_Node import Node
from graphviz import Digraph
import webbrowser
from graphviz import Digraph
import webbrowser
from collections import defaultdict
import data_parllel

def load_model(model_path):
    """Loads an ONNX model, performs shape inference, and prints input/output shapes for each layer."""
    model = onnx.load(model_path)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
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


def create_svg_graph(nodes_list, output_file="onnx_model_graph"):
    """Generate an SVG graph of the nodes using Graphviz and open it automatically."""

    dot = Digraph(comment='ONNX Model Graph', format='svg')

    # Add nodes with GPU number and styling
    for node in nodes_list:
        # Build label: name, op_type, and GPU number (if available)
        gpu_info = f"GPU {node.gpu_num}" if node.gpu_num is not None else "No GPU"
        label = f"{node.name}\n({node.op_type})\n{gpu_info}"
        dot.node(str(id(node)), label=label)

    # Add edges
    for node in nodes_list:
        for parent in node.parents:
            dot.edge(str(id(parent)), str(id(node)))

    # Render the graph to an SVG file
    out_path = dot.render(f"svg_file/{output_file}", view=False)  # view=False so we can open explicitly
    print(f"Graph saved as: {out_path}")

    # Automatically open the SVG in a new browser tab
    #webbrowser.open_new_tab(out_path)


# unused function
def create_svg_graph_with_clusters(nodes_list, output_file="clustered_graph"):
    """
    Generate an SVG graph using Graphviz where each GPU's nodes are grouped into a subgraph (cluster).
    Collective ops are placed outside the clusters.
    """

    dot = Digraph(comment="Clustered Data Parallel Graph", format='svg')

    # Split nodes by GPU
    from collections import defaultdict
    gpu_groups = defaultdict(list)
    for node in nodes_list:
        if node.collective:
            continue
        gpu_groups[node.gpu_num].append(node)

    # Add collective nodes globally
    for node in nodes_list:
        if node.collective:
            label = f"{node.name}\n({node.op_type})\nGPU {node.gpu_num}"
            dot.node(str(id(node)), label=label, shape="box", style="filled", fillcolor="lightblue")

    # Add clusters per GPU
    for gpu, group_nodes in gpu_groups.items():
        with dot.subgraph(name=f"cluster_gpu_{gpu}") as c:
            c.attr(label=f"GPU {gpu}")
            c.attr(style='rounded')
            for node in group_nodes:
                label = f"{node.name}\n({node.op_type})"
                c.node(str(id(node)), label=label)

    # Step 4: Add edges globally
    for node in nodes_list:
        for parent in node.parents:
            dot.edge(str(id(parent)), str(id(node)))

    # Render
    out_path = dot.render(f"svg_file/{output_file}", view=False)
    print(f"Clustered Graph saved to: {out_path}")
    webbrowser.open_new_tab(out_path)

def create_interactive_high_level_svg(model_replicas, output_file="interactive_high_level"):
    """
    Creates an interactive high-level SVG graph:
    - One node per GPU, linking to detailed GPU graph (e.g., gpu_0_detail.svg)
    - Shared collective ops: ScatterInput and AllReduceGrad
    - Clickable GPU nodes open their detailed SVGs
    """
    dot = Digraph(comment="High-Level Data Parallel Graph", format='svg')

    # Step 1: Flatten all nodes
    all_nodes = data_parllel.flatten_and_dedup(model_replicas)

    # Step 2: Separate collective ops and GPU-local ops
    gpu_groups = defaultdict(list)
    collectives = []
    for node in all_nodes:
            if node.collective:
                collectives.append(node)
            else:
                gpu_groups[node.gpu_num].append(node)

    # Step 3: Add collective nodes
    for collective_node in collectives:
        label = f"{collective_node.name}\n({collective_node.op_type})"
        dot.node(str(id(collective_node)), label=label, shape="box", style="filled", fillcolor="lightblue")

    # Step 4: Add abstract GPU nodes and connect to collectives
    for gpu, group_nodes in gpu_groups.items():
        gpu_node_name = f"gpu_{gpu}"
        label = f"GPU {gpu}\n(ReplicaModel)"
        href = f"gpu_{gpu}_detail.svg"

        dot.node(gpu_node_name, label=label, shape="box3d", style="filled", fillcolor="lightgray", href=href, target="_blank")

        # For each collective, check if it's linked to or from this replica
        for collective_node in collectives:
            for node in group_nodes:
                if collective_node in node.parents:
                    dot.edge(str(id(collective_node)), gpu_node_name)
                if node in collective_node.parents:
                    dot.edge(gpu_node_name, str(id(collective_node)))

    # Step 5: Render
    out_path = dot.render(f"svg_file/{output_file}", view=False)
    print(f"Interactive high-level SVG saved to: {out_path}")
    webbrowser.open_new_tab(out_path)