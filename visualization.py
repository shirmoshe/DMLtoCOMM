#!/usr/bin/env python3

# cd frontend
# npm install -D tailwindcss postcss autoprefixer
#  npx tailwindcss init -p
# npm.cmd start

"""
Render a provided root Node tree as a strict hierarchical DAG using Graphviz DOT,
ensuring all nodes at the same tree-depth appear on the same horizontal rank.
Uses cycle-safe DFS to collect all reachable nodes and edges, and a true longest-path
depth computation to place nodes correctly.
Outputs an SVG and opens it automatically for immediate viewing.

Usage:
  pip install graphviz
  from visualize_tree_dot import visualize_tree_dot
  visualize_tree_dot(root_node, main_d, main_t, main_p, total_gpu, output_base='parallel_full_graph')
"""

from graphviz import Digraph
from class_Node import Node
from typing import List, Tuple, Dict
import os
import webbrowser


def format_index(val: int) -> str:
    """
    Return "*" if val == 0, otherwise return the string representation of val.
    This ensures any 0 in a triplet is printed as "*".
    """
    return "*" if val == 0 else str(val)


def collect_from_root(root: Node) -> Tuple[List[Node], List[Tuple[int, int]]]:
    """
    Perform cycle-safe DFS from the given root to collect nodes and edges.
    Returns:
      nodes: list of Node (each visited once)
      edges: list of (parent_index, child_index)
    """
    seen = set()
    nodes: List[Node] = []
    edges: List[Tuple[int, int]] = []

    def dfs(n: Node):
        if n.index in seen:
            return
        seen.add(n.index)
        nodes.append(n)
        for c in n.children:
            edges.append((n.index, c.index))
            dfs(c)

    dfs(root)
    return nodes, edges


def compute_depths(root: Node) -> Dict[int, int]:
    """
    Compute true longest root->node path depth for each node.
    Depth(root)=0; depth(n)=1+max(depth(p) for p in n.parents).
    Returns mapping node_index -> depth.
    """
    nodes_list, _ = collect_from_root(root)
    depth_map: Dict[int, int] = {}

    def dfs_depth(n: Node) -> int:
        if n.index in depth_map:
            return depth_map[n.index]
        if not n.parents or n == root:
            depth_map[n.index] = 0
        else:
            parent_depths = [dfs_depth(p) for p in n.parents]
            depth_map[n.index] = max(parent_depths) + 1
        return depth_map[n.index]

    for n in nodes_list:
        dfs_depth(n)
    return depth_map


def get_short_edge_labels(op_type: str) -> Tuple[str, str]:
    """
    Return (label_for_incoming_edge, label_for_outgoing_edge)
    for the given collective operator type, using short concise labels.
    """

    mapping = {
        "AllGather_TP":      ("",     ""),
        "AllReduce_TP":      ("",     ""),
        "ReduceScatter_TP":  ("",     ""),
        "AllReduce_DP":      ("",     ""),
        "P2P":               ("",     ""),
    }
    return mapping.get(op_type, ("", ""))

def _format_shape(shape: tuple | list) -> str:
    """Return 'd1×d2×…' for an iterable of ints."""
    return "×".join(map(str, shape))


def build_shape_info(n: Node) -> str:
    """
    Return the compact string to show in the bottom row:
        "<num_reps>·(<out_shape>)"
    Skip for op_types that shouldn't show shapes.
    """
    if n.op_type in {"READ_DATA_DP", "AllReduce_DP"}:
        return ""

    if n.shape_out:
        out_shape = "×".join(map(str, n.shape_out))
        reps = f"{n.num_reps_inside}·" if n.num_reps_inside > 1 else ""
        return f"{reps}({out_shape})"
    return ""

def make_html_label(
    parsed_name_label: str,
    node_type_label: str,
    note_text: str,
    gpu_entries: List[str],
    bgcolor: str,
    arrange_vertical: bool,
    round_outer: bool,
    shape_info: str = "",           # ← new optional column (right-hand side)
) -> str:
    """
    Build an HTML-like label for a Graphviz node.

    Structure produced
    ------------------
    ┌───────────────────────── outer TABLE ─────────────────────────┐
    │  Row-1  ─ node_type_label  |  parsed_name_label | [shape_info]│
    │  Row-2  ─ (optional) italic note spanning two columns         │
    │  Row-3  ─ "Assigned GPU(s):" | inner TABLE with GPU entries   │
    └───────────────────────────────────────────────────────────────┘

    The right-hand cell [shape_info] is shown only when *shape_info*
    is non-empty.  It spans all rows via ROWSPAN so it always aligns
    vertically with the rest of the node content.
    """

    # ------------------------------------------------------------------
    # 1) Header text for the GPU section
    # ------------------------------------------------------------------
    header_text = "Assigned GPU:" if len(gpu_entries) == 1 else "Assigned GPU's:"

    # ------------------------------------------------------------------
    # 2) Build the inner TABLE that lists GPU entries
    # ------------------------------------------------------------------
    inner_rows: List[str] = []
    if arrange_vertical:
        # One GPU entry per row
        for entry in gpu_entries:
            inner_rows.append(
                f'<TR><TD BORDER="1" STYLE="ROUNDED" CELLPADDING="4">{entry}</TD></TR>'
            )
    else:
        # All GPU entries in a single row
        cells = [
            f'<TD BORDER="1" STYLE="ROUNDED" CELLPADDING="4">{entry}</TD>'
            for entry in gpu_entries
        ]
        inner_rows.append("<TR>\n" + "\n".join(cells) + "\n</TR>")

    inner_table = (
        '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4">\n'
        + "\n".join(inner_rows)
        + "\n</TABLE>"
    )

    # ------------------------------------------------------------------
    # 3) Determine outer style and ROWSPAN for the shape column
    # ------------------------------------------------------------------
    outer_style = ' STYLE="ROUNDED"' if round_outer else ""
    rowspan = 3 if not note_text else 4  # extra row if we have a note

    # Right-hand cell (only if shape_info provided)

    right_td = (
        f'''
        <!-- thin grey divider + content cell -->
        <TD WIDTH="1" BGCOLOR="#B0B0B0" ROWSPAN="{rowspan}"></TD>
        <TD ALIGN="CENTER" CELLPADDING="4" ROWSPAN="{rowspan}">
            {shape_info}
        </TD>
        '''
    ) if shape_info else ""


    # ------------------------------------------------------------------
    # 4) Assemble rows of the outer TABLE
    # ------------------------------------------------------------------
    html_rows: List[str] = []

    # Row-1 : title + parsed name (+ optional shape column)
    html_rows.append(
        f'''
  <TR>
    <TD ALIGN="CENTER" CELLPADDING="4"><B>{node_type_label}</B></TD>
    <TD ALIGN="CENTER" CELLPADDING="4">{parsed_name_label}</TD>
    
  </TR>'''
    )

    # Row-2 : optional note (italic, spans two columns)
    if note_text:
        html_rows.append(
            f'''
  <TR>
    <TD COLSPAN="2" ALIGN="CENTER" CELLPADDING="4"><I>{note_text}</I></TD>
  </TR>'''
        )

    # Row-3 : GPU header + nested TABLE of GPU entries
    html_rows.append(
        f'''
  <TR>
    <TD ALIGN="CENTER" CELLPADDING="4">{header_text}</TD>
    <TD ALIGN="CENTER" CELLPADDING="4">
      {inner_table}
    </TD>
  </TR>'''
    )
    # Row-4 : bottom row for shape info (if any)
    if shape_info:
        html_rows.append(f'''
    <TR>
      <TD ALIGN="LEFT" CELLPADDING="4"><I>shape:</I></TD>
      <TD ALIGN="LEFT" CELLPADDING="4">{shape_info}</TD>
    </TR>''')

    # ------------------------------------------------------------------
    # 5) Wrap everything in the outer TABLE and return
    # ------------------------------------------------------------------
    html_label = (
        f'''<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="4" '''
        f'''BGCOLOR="{bgcolor}" CELLPADDING="6"{outer_style}>'''
        + "\n"
        + "\n".join(html_rows)
        + "\n</TABLE>>"
    )
    return html_label



def visualize_tree_dot(root: Node,
                       main_d: int,
                       main_t: int,
                       main_p: int,
                       output_base: str = 'graph') -> None:
    """
    :param root: root Node of the DAG
    :param main_d: number of data-parallel partitions
    :param main_t: number of tensor-parallel partitions
    :param main_p: number of pipeline-parallel partitions
    :param total_gpu: total number of GPUs available
    :param output_base: base filename for the output SVG
    """

    # 1) Collect all nodes and edges via DFS
    nodes_list, edges = collect_from_root(root)

    # Build a map from index to Node for quick lookup
    index_to_node = {n.index: n for n in nodes_list}

    # 2) Compute depth of each node so we can enforce same-rank layers
    depth_map = compute_depths(root)

    # 3) Initialize a Graphviz Digraph
    dot = Digraph('G', format='svg')
    dot.attr(rankdir='TB', splines='polyline')

    # 3a) Add a legend explaining what "*" means.
    # Place it at the top by giving it the minimum rank.
    legend_label = f" {main_d} {main_t} {main_p}"
    dot.node('legend', label=legend_label, shape='none')
    with dot.subgraph() as lg:
        lg.attr(rank='min')
        lg.node('legend')

    # 4) Style and define each node
    for n in nodes_list:
        # Use format_index to replace any 0 with "*"
        n_d_str = format_index(n.id_d)
        n_p_str = format_index(n.id_p)
        n_t_str = format_index(n.id_t)

        # Determine fill color and labels by node_type and layer parity
        n_node_type = n.node_type.lower()
        if n_node_type == 'compute':
            fill = 'lightyellow1' if (n.layer % 2 == 0) else 'lightyellow'
            arrange_vertical = False
            round_outer = False  # no capsule for compute nodes

        elif n_node_type == 'collective':
            fill = 'lightgreen'
            arrange_vertical = False
            round_outer = True   # capsule shape for collective nodes

        elif n_node_type == 'local_compute':
            fill = 'lightblue'
            arrange_vertical = False
            round_outer = True

        elif n_node_type == 'p2p':
            fill = 'pink'
            arrange_vertical = True    # GPU entries stacked vertically
            round_outer = True         # capsule shape for P2P nodes

        else:
            fill = 'lightgray'
            arrange_vertical = False
            round_outer = False

        # Prepare parsed_name by replacing "/" with "|" etc.
        parsed_name = n.name.replace('/', '  |  ').replace('layers.', 'Layer: ')

        node_type_label = f"{n.node_type}:"  # full node_type as title

        # Determine note_text based on node_type or op_type
        note_text = ""
        if "MatMul_Q4" in n.name :
            note_text = "(sharded by columns)"
        elif n.op_type == "GroupQueryAttention":
            note_text = "(two matrix multiplications - pair wise sharding)"

        # Prepare the list of GPU entries
        gpu_entries: List[str] = []
        if n.is_tp_candidate:
            # Multiple GPU entries: one entry per tensor-parallel partition
            for t in range(1, main_t + 1):
                d_part = n_d_str
                t_part = format_index(t)
                p_part = n_p_str
                gpu_entries.append(f"[{d_part},{t_part},{p_part}]")
        elif n.op_type.lower() == 'p2p':
            # Replace multiple entries with a single alpha‐based representation:
            d_part = n_d_str
            alpha = "α"
            p_part = n_p_str
            next_p = n.id_p + 1
            next_p_str = format_index(next_p)
            gpu_entries.append(f"[{d_part},{alpha},{p_part}] → [{d_part},{alpha},{next_p_str}]")
        else:
            # Single GPU entry
            gpu_entries.append(f"[{n_d_str},{n_t_str},{n_p_str}]")

        shape_info = build_shape_info(n)

        # Build the final HTML-like label with optional rounded outer table
        html_label = make_html_label(
            parsed_name_label=parsed_name,
            node_type_label=node_type_label,
            note_text=note_text,
            gpu_entries=gpu_entries,
            bgcolor=fill,
            arrange_vertical=arrange_vertical,
            round_outer=round_outer,
            shape_info=shape_info
        )

        # Add the node with the HTML label; shape='none' so no extra bounding box
        dot.node(
            name=str(n.index),
            label=html_label,
            shape='none'
        )

    # 5) Enforce same rank per tree depth
    max_depth = max(depth_map.values(), default=0)
    for depth in range(max_depth + 1):
        with dot.subgraph() as sg:
            sg.attr(rank='same')
            for n in nodes_list:
                if depth_map.get(n.index, 0) == depth:
                    sg.node(str(n.index))

    # 6) Draw edges between nodes, adding short labels for collective transitions
    for src_idx, tgt_idx in edges:
        src_node = index_to_node[src_idx]
        tgt_node = index_to_node[tgt_idx]

        # If target is a collective node → label incoming edge
        if tgt_node.node_type.lower() == 'collective':
            op_type = tgt_node.op_type  # e.g., "AllGather_TP", "ReduceScatter_TP", etc.
            incoming_label, _ = get_short_edge_labels(op_type)
            if incoming_label:
                dot.edge(str(src_idx), str(tgt_idx), label=incoming_label)
            else:
                dot.edge(str(src_idx), str(tgt_idx))

        # Else if source is a collective node → label outgoing edge
        elif src_node.node_type.lower() == 'collective':
            op_type = src_node.op_type
            _, outgoing_label = get_short_edge_labels(op_type)
            if outgoing_label:
                dot.edge(str(src_idx), str(tgt_idx), label=outgoing_label)
            else:
                dot.edge(str(src_idx), str(tgt_idx))

        # Otherwise, no label
        else:
            dot.edge(str(src_idx), str(tgt_idx))

    # 7) Render to SVG and open in a browser
    svg_path = dot.render(filename=output_base, cleanup=True)
    if not svg_path.endswith('.svg'):
        svg_path += '.svg'
    print(f"SVG rendered to: {svg_path}")
    webbrowser.open_new_tab('file://' + os.path.abspath(svg_path))
