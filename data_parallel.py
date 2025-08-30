
from typing import List, Tuple, Dict
from collections import deque

from class_Node import Node


def get_firsts(nodes_list: List[Node]) -> List[Node]:

    first_nodes = []
    for node in nodes_list[:]:
        if not node.parents:
            #print(f"First: {node.name}")
            first_nodes.append(node)
    return first_nodes


def get_lasts(nodes_list: List[Node]) -> List[Node]:

    last_nodes = []
    for node in nodes_list[:]:
        if not node.children:
            #print(f"Last: {node.name}")
            last_nodes.append(node)
    return last_nodes

def replicate_by_root(root: Node, d: int) -> List[Node]:
    """
    Given a single root, produce d “replicas” of its entire subtree.
    The original tree remains with id_d=1; each new replica gets id_d=2..d.
    Returns a flat list of all nodes (original + replicas).
    """
    # 1) Traverse the original tree to get a list of nodes in BFS order:
    originals: List[Node] = []
    queue = deque([root])
    seen = set([root.index])
    while queue:
        n = queue.popleft()
        originals.append(n)
        for c in n.children:
            if c.index not in seen:
                seen.add(c.index)
                queue.append(c)

    # 1a) force originals to id_d = 1
    for orig in originals:
        orig.id_d = 1

    # 2) For each replica index, deep‐clone every node and reset id_d
    all_clones: List[Node] = []
    for d_idx in range(2, d + 1):
        clone_map: Dict[int, Node] = {}
        # First pass: clone node objects without linking
        for orig in originals:
            clone = orig.clone()   # assume Node.clone() duplicates name, op_type, layer, etc.
            clone.id_d = d_idx
            # clear out any existing parent/child pointers on the clone
            clone.parents = []
            clone.children = []
            clone_map[orig.index] = clone
            all_clones.append(clone)

        # Second pass: rebuild parent→child links among clones
        for orig in originals:
            clone = clone_map[orig.index]
            for orig_child in orig.children:
                child_clone = clone_map[orig_child.index]
                # link clone → child_clone (and child_clone ← clone under the hood)
                clone.add_parent_child_link(child_clone)

    # 3) Return the concatenation: original nodes + all their clones
    return originals + all_clones




def warp_by_new_data_collectives(nodes_list : list[Node]) -> list[Node]:

    first_nodes = get_firsts(nodes_list)
    last_nodes  = get_lasts(nodes_list)

    # 2) create Scatter and AllReduceGrad for the warp

    ScatterInput = Node(
        name="READ_DATA",
        op_type="READ_DATA_DP",
        layer=0)

    ScatterInput.node_type = "local_compute"
    ScatterInput.id_d = 0
    ScatterInput.id_t = 0
    ScatterInput.id_p = 0
    ScatterInput.shape_in = [1024, 1024]

    AllReduceGrad = Node(
        name="AllReduceGrad",
        op_type="AllReduce_DP",
        layer=0)

    AllReduceGrad.node_type = "collective"
    AllReduceGrad.id_d = 0
    AllReduceGrad.id_t = 0
    AllReduceGrad.id_p = 0

    # 3) linking collective nodes to the trees


    # add ScatterInput <-> first_nodes connection
    for node in first_nodes:
        ScatterInput.add_parent_child_link(node)

    # add allgather <-> last_nodes connection
    for node in last_nodes:
        node.add_parent_child_link(AllReduceGrad)


    return [ScatterInput] + nodes_list + [AllReduceGrad]

def apply_data_parallel(nodes_list: List[Node], data_size: int, d: int) -> Tuple[List[Node], Node]:
    # 1) find the root of your DP tree:

    root = get_firsts(nodes_list)[0]

    # 2) replicate the entire subtree under root:
    replicated = replicate_by_root(root, d)

    # 3) insert collectives around that replicated set:
    with_collectives = warp_by_new_data_collectives(replicated)

    # 4) scatter is the first element of with_collectives
    scatter = with_collectives[0]

    # 5) return the full list plus the scatter node
    return with_collectives, scatter

