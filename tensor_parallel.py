from class_Node import Node
from typing import List, Tuple, Dict


def is_tp_node(node: Node) -> bool:
    """Detect nodes that belong to a tensor-parallel block."""
    name = node.name
    node_type = node.node_type
    if ("MatMul_Q4" in name or "GroupQueryAttention" in name) and node_type == "compute":
        node.is_tp_node = True
        return True
    else:
        node.is_tp_node = False
        return False


def warp_by_tp_collectives(node) -> tuple[Node, Node]:
    # 1) get node original parent and child connections
    parents_list = list(node.parents)
    children_list = list(node.children)

    # 2) create Scatter and AG/AR for the warp
    local_chank = Node(
        name=f"local_chunking{node.name}",
        op_type="local_chunking",
        layer=node.layer
    )
    local_chank.node_type = "local_compute"
    local_chank.id_d = getattr(node, "id_d", 0)
    local_chank.id_t = getattr(node, "id_t", 0)
    local_chank.id_p = getattr(node, "id_p", 0)
    local_chank.shard_config = "COL"

    if ("MatMul_Q4" or "MatMulNBits") in node.name:
        collect_collective = Node(
            name=f"layer.{node.layer}/AllGather_TP",
            op_type="AllGather_TP",
            layer=node.layer
        )
        collect_collective.node_type = "collective"
        collect_collective.id_d = getattr(node, "id_d", 0)
        collect_collective.id_t = getattr(node, "id_t", 0)
        collect_collective.id_p = getattr(node, "id_p", 0)
    else:
        collect_collective = Node(
            name=f"layer.{node.layer}/AllReduce_TP",
            op_type="AllReduce_TP",
            layer=node.layer
        )
        collect_collective.node_type = "collective"
        collect_collective.id_d = getattr(node, "id_d", 0)
        collect_collective.id_t = getattr(node, "id_t", 0)
        collect_collective.id_p = getattr(node, "id_p", 0)

    # 3) linking collective nodes to the tree

    # delete any old connections from node and family
    for parent in parents_list:
        parent.remove_child(node)
        node.remove_parent(parent)
    for child in children_list:
        child.remove_parent(node)
        node.remove_child(child)

    #add parent <-> local_chank connection
    for parent in parents_list:
        parent.add_child(local_chank)
        local_chank.add_parent(parent)

    # add local_chank <-> node connection
    local_chank.add_child(node)
    node.add_parent(local_chank)


    # add node <-> collect_collective
    node.add_child(collect_collective)
    collect_collective.add_parent(node)

    # add collect_collective <-> children
    for child in children_list:
        collect_collective.add_child(child)
        child.add_parent(collect_collective)

    return local_chank, collect_collective


def delete_scatters(node_list: List[Node]) -> List[Node]:
    """
    Remove any Scatter_COL nodes that have a parent collective of type AllGather_TP or AllReduce_TP.
    Reconnect their children directly to that parent to maintain connectivity, then remove the local_chunking.
    """
    to_remove = []
    for node in list(node_list):
        # Identify local_chunking nodes by op_type
        if node.op_type == "local_chunking":
            # Find any parent whose op_type is "AllGather_TP" or "AllReduce_TP"
            valid_parents = [
                p for p in node.parents
                if p.op_type in ("AllGather_TP", "AllReduce_TP") or p.node_type is not None     #delete all local_chunking
            ]
            if valid_parents:
                # Assume a single valid parent; reconnect children to that parent
                parent_collective = valid_parents[0]
                for child in list(node.children):
                    # Remove node <-> child
                    node.remove_child(child)
                    child.remove_parent(node)
                    # Add parent_collective <-> child
                    parent_collective.add_child(child)
                    child.add_parent(parent_collective)
                # Remove parent_collective <-> node
                parent_collective.remove_child(node)
                node.remove_parent(parent_collective)
                # Mark local_chunking for removal
                to_remove.append(node)

    # Remove all marked local_chunking nodes from the node_list
    for local_chunking in to_remove:
        if local_chunking in node_list:
            node_list.remove(local_chunking)

    return node_list


def apply_tensor_parallel(node_list: List[Node], t_size: int = 0) -> List[Node]:
    """
    Identify tensor-parallel candidate nodes, insert appropriate collective nodes (scatter + gather/reduce),
    then delete any scatter nodes whose parent is AllGather_TP or AllReduce_TP,
    reconnecting their children to the parent collective.
    """
    new_collective_nodes: List[Node] = []
    for node in node_list:
        node.is_tp_candidate = is_tp_node(node)
        if node.is_tp_candidate:
            # warp_by_tp_collectives returns (scatter_node, collect_node)
            scatter_node, collect_node = warp_by_tp_collectives(node)
            # add both to our temp list
            new_collective_nodes.extend([scatter_node, collect_node])

    # Extend the original list with the newly created collective nodes
    extended_list = node_list + new_collective_nodes
    # Now remove any redundant scatter nodes and reconnect their children
    cleaned_list = delete_scatters(extended_list)
    return cleaned_list
