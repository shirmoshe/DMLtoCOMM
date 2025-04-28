from copy import deepcopy
from class_Node import Node
from collections import defaultdict
import result_visualization

# take a parameter d from the main.
# add node to the graph at the beginnig:
## the node is collective operator, that split the data for all the d GPU
## then each gpu from the d GPU has a replica of the whole model


def replicate_model(nodes_list: list[Node], d: int) -> list[list[Node]]:
    """
      Replicates the original ONNX node graph d times for data parallelism.
      Each replica is a full copy of the model with renamed nodes and corrected parent relationships.

      Args:
          nodes_list (list[Node]): List of original ONNX nodes.
          d (int): Number of model replicas (e.g., number of GPUs).

      Returns:
          list[list[Node]]: A list of d replicas, each being a list of Node objects.
      """

    replicas = []

    for i in range(d):
        node_mapping = {}  # Maps original_node -> cloned_node for this replica
        replica_nodes = []

        # Clone each node and rename it for replica i
        for node in nodes_list:
            cloned_node = Node(
                index=node.index,
                name=node.name,
                op_type=node.op_type,
                layer=node.layer
            )
            cloned_node.id_d = i
   #         cloned_node.gpu_num = i
            node_mapping[node] = cloned_node
            replica_nodes.append(cloned_node)

        # Reconnect parents using cloned nodes
        for original_node in nodes_list:
            cloned_node = node_mapping[original_node]
            cloned_node.parents = [node_mapping[parent] for parent in original_node.parents]

        replicas.append(replica_nodes)

    return replicas

def create_data_parallel_collectives(node_list, d, data_size):
    """
       Creates collective communication nodes (Scatter and AllReduce)
       for data parallelism and connects them to the model replicas.

       Args:
           node_replicas (list[list[Node]]): A list of model replicas, each as a list of Node objects.

       Returns:
           list[Node]: A flat list of all nodes including replicas and collective operators.
       """
    batch_size_per_replica = data_size / d


    node_replicas = replicate_model(node_list, d)

    full_replicas = []

    allreduce_node = Node(index="data_collective_1", name="AllReduceGrad", op_type="AllReduce")
    allreduce_node.gpu_num = 0
    allreduce_node.collective = True

    # Connect Read → Input node of each replica
    for i, replica_nodes in enumerate(node_replicas):
        # create read data nodes
        read_data_node = Node(index=f"data_read_{i:02d}", name="ReadData", op_type="Read")
        read_data_node.gpu_num = i
        node_replicas[i].append(read_data_node)

        input_nodes = [n for n in replica_nodes if len(n.parents) == 0]  # find nodes with no parents = input nodes
#        if len(input_nodes) != 1:
#           raise ValueError("Each replica must have exactly one input node.")
        input_node = input_nodes[0]
        input_node.parents.append(read_data_node)
        read_data_node.data_size = batch_size_per_replica

    # Connect last node of each replica → AllReduce
    for replica_nodes in node_replicas:
        output_nodes = [n for n in replica_nodes if not any(n in other.parents for other in replica_nodes)]  # find nodes that no one points to (leaf/output nodes)
#        if len(output_nodes) != 1:
#            raise ValueError("Each replica must have exactly one output node.")
        output_node = output_nodes[0]
        allreduce_node.parents.append(output_node)

        # Build the full list for this replica
        full_list = replica_nodes + [allreduce_node]
        full_replicas.append(full_list)

    return full_replicas


def flatten_and_dedup(full_replicas):
    """
    Flattens the list of replicas into a single list of unique Node objects.
    This avoids duplicating shared collective nodes like Scatter and AllReduce.
    """
    seen = set()
    merged = []

    for replica in full_replicas:
        for node in replica:
            if id(node) not in seen:
                merged.append(node)
                seen.add(id(node))

    return merged


def create_data_parallel(nodes_list, d, data_size):
    model_replicas = create_data_parallel_collectives(nodes_list, d, data_size)  # replica model d times
    result_visualization.create_interactive_high_level_svg(model_replicas)      # create interactive high level graph
    for d_idx, replica in enumerate(model_replicas):     # generate each detailed replica view
        result_visualization.create_svg_graph(replica, output_file=f"data_{d_idx}_detail")