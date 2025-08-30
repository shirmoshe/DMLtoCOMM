
from typing import List
from class_Node import Node
from typing import List, Tuple, Dict


def assign_pipeline_stages(nodes_list: List[Node], p: int) -> List[Node]:
    """
    Assign pipeline stage IDs (1..p) by slicing the model's layers into p consecutive groups, using -1 for undefined layers.
    Each stage corresponds to a contiguous block of layers.

    Args:
        nodes_list (List[Node]): Flat list of Node instances with .layer set.
        p (int): Total number of pipeline stages.

    Returns:
        List[Node]: The same list with each node.id_p set to 0..p-1, or -1 if layer<0.
    """
    # Collect sorted unique layers (ignore negative)
    layers = sorted({n.layer for n in nodes_list if n.layer >= 0})
    total_layers = len(layers)

    # If no valid layers, mark all nodes as undefined
    if total_layers == 0:
        for n in nodes_list:
            n.id_p = -1
        return nodes_list

    # Validate stage count
    if p <= 0:
        raise ValueError("Number of pipeline stages p must be positive")

    # Compute base number of layers per stage and extra remainder
    base, extra = divmod(total_layers, p)

    # Build a flat list mapping each layer index to a 1-based stage ID
    stage_map: List[int] = []
    for stage_idx in range(p):
        count = base + (1 if stage_idx < extra else 0)
        # stage_idx + 1 gives a 1-based stage number
        stage_map.extend([stage_idx + 1] * count)

    # Map each actual layer value to its assigned 1-based stage
    layer_to_stage = {layers[i]: stage_map[i] for i in range(total_layers)}

    # Assign stage ID to each node
    for n in nodes_list:
        n.id_p = layer_to_stage.get(n.layer, -1)
        if n.node_type == "collective" and n.name in ("READ_DATA", "AllReduceGrad"):
            n.id_p = 0

    return nodes_list

def insert_pipeline_p2ps(nodes_list: List[Node]) -> List[Node]:
    """
    Insert at most one P2P node between each (parent.id_p, child.id_p, id_d, id_t) group.
    Each P2P is merged across all parents for the same stage transition.
    """
    new_nodes = []
    p2p_map = {}  # key: (parent.id_p, child.id_p, id_d, id_t), value: p2p_node

    for child in nodes_list:
        new_parents = []
        # Build mapping from (src stage) -> [parents]
        parent_stage_map = {}
        for parent in child.parents:
            if 0 < parent.id_p != child.id_p > 0:

                key = (parent.id_p, child.id_p, child.id_d, child.id_t, parent.layer)
                parent_stage_map.setdefault(key, []).append(parent)
            else:
                new_parents.append(parent)
        # For each stage transition, create or reuse P2P
        for key, parent_list in parent_stage_map.items():
            (p_src, p_dst, id_d, id_t, p_layer) = key
            if key not in p2p_map:
                p2p = Node(
                    name=f"P2P stage {p_src} to stage {p_dst}",
                    op_type="P2P",
                    layer= p_layer,)
                p2p.node_type = "P2P"
                p2p.id_d = id_d
                p2p.id_t = id_t
                p2p.id_p = p_dst
                p2p.p_src = p_src
                p2p.p_dst = p_dst
                p2p.parents = parent_list.copy()
                p2p.children = [child]
                new_nodes.append(p2p)
                p2p_map[key] = p2p
                # Update parent's children to point to P2P instead of child
                for parent in parent_list:
                    if child in parent.children:
                        parent.children = [p2p if c is child else c for c in parent.children]
                    else:
                        parent.children.append(p2p)
            else:
                p2p = p2p_map[key]
                for parent in parent_list:
                    if parent not in p2p.parents:
                        p2p.parents.append(parent)
                    if child in parent.children:
                        parent.children = [p2p if c is child else c for c in parent.children]
                    else:
                        parent.children.append(p2p)
                if child not in p2p.children:
                    p2p.children.append(child)
            # Replace all such parents in the child's parent list with the P2P node
            # (if multiple, only once!)
            if p2p not in new_parents:
                new_parents.append(p2p)
        child.parents = new_parents

    #print(f"Total merged P2P nodes inserted: {len(new_nodes)}")
    return nodes_list + new_nodes

def apply_pipeline_parallel(nodes_list: List[Node], p: int) -> List[Node]:
    """
    Full pipeline-parallel pass: assign stage IDs then insert P2P nodes.
    """
    # assign id_p based on existing .layer values
    staged = assign_pipeline_stages(nodes_list, p)
    # insert P2P send/recv at boundaries

    #fix data paralel nodes #patch
    for n in staged:
        if "READ_DATA" in n.name or "AllReduceGrad" in n.name:
            n.id_p = 0
            #print(n.name + " p is now 0)")
    return insert_pipeline_p2ps(staged)

