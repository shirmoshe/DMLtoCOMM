import onnx
from class_Node import Node
from graphviz import Digraph
import webbrowser
from graphviz import Digraph
import webbrowser
from collections import defaultdict
import data_parllel
import os


def create_svg_graph(nodes_list, output_file="onnx_model_graph"):
    """Generate an SVG graph of the nodes using Graphviz and open it automatically."""

    dot = Digraph(comment='ONNX Model Graph', format='svg')

    # Add nodes with parameters (id d p t, data size)
    for node in nodes_list:
        # Build label: name, op_type
#        gpu_num = f"data id {node.id_d}" if node.id_d is not None else "No GPU"
        label = f"data id: {node.id_d}\nlayer: {node.layer}\n{node.name}\n({node.op_type})\ndata_size: {node.data_size}"
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


def create_svg_graph_with_clusters(nodes_list, output_file="clustered_graph"):
    """
    Generate an SVG graph using Graphviz where each GPU's nodes are grouped into a subgraph (cluster).
    Collective ops are placed outside the clusters.
    """

    dot = Digraph(comment="Clustered Data Parallel Graph", format='svg')

    # Split nodes by id_d
    from collections import defaultdict
    gpu_groups = defaultdict(list)
    for node in nodes_list:
        if node.collective:
            continue
        gpu_groups[node.gpu_num].append(node)

    # Add collective nodes globally
    for node in nodes_list:
        if node.collective:
            label = f"{node.name}\n({node.op_type}\ndata id: {node.id_d})"
            dot.node(str(id(node)), label=label, shape="box", style="filled", fillcolor="lightblue")

    # Add clusters per data
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
    DATA Parallelism
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
                gpu_groups[node.id_d].append(node)

    # Step 3: Add collective nodes
    for collective_node in collectives:
        label = f"{collective_node.name}\n({collective_node.op_type})"
        dot.node(str(id(collective_node)), label=label, shape="box", style="filled", fillcolor="lightblue")

    # Step 4: Add abstract GPU nodes and connect to collectives
    for gpu, group_nodes in gpu_groups.items():
        gpu_node_name = f"gpu_{gpu}"
        label = f"Data id: {gpu}\n(ReplicaModel)"
        href = f"data_{gpu}_detail.svg"

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


def create_layered_svg(layer_groups, d_id=0, output_dir="svg_file"):
    """
    create high level interactive graph with layers
    """
    # make sure the folder exist, if not - open a new one
    os.makedirs(output_dir, exist_ok=True)

    dot = Digraph(comment=f"Layered Graph for data replica {d_id}", format='svg')
    dot.attr(rankdir='TB', compound='true')

    for layer_idx, nodes in layer_groups.items():
        box_id = f"layer_{d_id}_{layer_idx}"
        label = f"layer {layer_idx}"
        href = f"layer_{layer_idx}_data_{d_id}_detail.svg"
        dot.node(box_id, label=label, shape="box", style="filled", fillcolor="lightyellow", href=href, target="_blank")

        # create subgraph for each layer
        save_layer_detail_svg(nodes, layer_idx, d_id, output_dir)

    # add global edges between layers
    added_edges = set()

    for target_layer, nodes in layer_groups.items():
        for node in nodes:
            for parent in node.parents:
                source_layer = getattr(parent, 'layer', -1)
                if source_layer != target_layer and source_layer != -1:  # don't include layer -1
                    if source_layer != target_layer:
                        src = f"layer_{d_id}_{source_layer}"
                        tgt = f"layer_{d_id}_{target_layer}"
                        if (src, tgt) not in added_edges:
                            dot.edge(src, tgt)
                            added_edges.add((src, tgt))

    out_path = dot.render(filename=f"{output_dir}/data_{d_id}_detail", view=False)
    print(f"Graph saved as: {output_dir}/data_{d_id}_detail")


def save_layer_detail_svg(layer_nodes, layer_idx, d_id, output_dir="svg_file"):
    """
    save layer's subgraph
    """
    dot = Digraph(comment=f"Layer {layer_idx} data {d_id} Detail", format="svg")

    for node in layer_nodes:
        label = f"{node.name}\n({node.op_type}\nlayer: {node.layer})"
        dot.node(str(id(node)), label=label)

    for node in layer_nodes:
        for parent in node.parents:
            if parent in layer_nodes:
                dot.edge(str(id(parent)), str(id(node)))

    filename = f"{output_dir}/layer_{layer_idx}_data_{d_id}_detail"
    dot.render(filename=filename, view=False)
    print(f"Graph saved as: {filename}.svg")
