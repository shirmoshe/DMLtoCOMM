from typing import List, Dict, Optional, Tuple
import math
import json
from pathlib import Path

import onnx_analyze
import data_parallel
import os
import pipeline_parallel
import tensor_parallel
import visualization

from class_Node import Node
from typing import List

global d, t, p, total_gpu

def main():



    # ------------------ INITIALIZATION ------------------
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path   = os.path.join(project_root, "load_model", "tiny_llama_model", "tiny_llama.onnx")
    json_path    = os.path.join(project_root, "load_model", "code_files", "user_inputs.json")


    onnx_model = onnx_analyze.load_model(model_path)
    config       = onnx_analyze.load_config(json_path)
    data_size    = onnx_analyze.get_model_data_size(config)

    # #validate paths
    # print("Loaded config:", config)
    # print("Topology keys:", config.get("topology", {}).keys())

    # visualization
    show_config_tree = tuple(config["visualization"]["show_config_tree"])

    # topology & hardware
    total_gpu        = config["topology"]["total_gpu"]
    bandwidth   = config["topology"]["bandwidth_gbps"]
    beta             = 1 / bandwidth
    flop_rate        = config["hardware"]["flop_rate"]

    # training
    N = config["training"]["dataset_size"]
    batch_size = config["training"]["global_batch_size"]
    num_mirco_batches_per_batch = config["training"]["micro_batches_per_batch"]

    num_of_batches = math.ceil((N / batch_size))
    #b = B // num_mirco_batches_per_batch #micro-batch size
    mirco_batch_size = batch_size // num_mirco_batches_per_batch

    min_t        = 2
    max_p        = 20           # num_layers -1
    # data parameters

    if mirco_batch_size < 1:
        print("mirco_batch_size must be greater than 1")


    L = 640    # Sequence length – number of tokens (sub-words) in **each** sequence.
    D= 640     #Hidden size – embedding / model dimension for **each** token.

    results = []

    # ------------------ ANALYZE EACH CONFIG ------------------
    for d, t, p in get_all_parallel_configs(total_gpu, min_t, max_p):  # d,t,p are  DP / TP / PP degrees per configuration

        # 1) build fresh tree
        nodes = onnx_analyze.create_nodes(onnx_model)
        nodes = clear_nodes_list_redundancy(nodes)
        nodes = clear_nodes_names(nodes)
        nodes_copy, main_root = data_parallel.apply_data_parallel(nodes.copy(), N, d)
        nodes_copy           = pipeline_parallel.apply_pipeline_parallel(nodes_copy, p)
        nodes_copy           = tensor_parallel.apply_tensor_parallel(nodes_copy)

        # 2) pre-graph analysis
        main_root.shape_in = [mirco_batch_size * d , L, D]
        main_root.rec_data_shape_flow(d, t, p)

        num_rounds =  math.ceil(num_of_batches / d)

        # 3) compute times & transfers
        if p == 1:
            # Full model sits on one stage: use the all-layers timing helper
            tau = main_root.calc_all_compute_time(flop_rate, t)  # sec / µ-batch
            step_time = num_mirco_batches_per_batch * tau  # steady flow only
        else:
            # Pipeline with p>1 stages: use stage-1 timing and add fill latency
            tau = main_root.calc_stage1_compute_time(flop_rate, t, p)  # sec / µ-batch / stage-1
            step_time = (num_mirco_batches_per_batch + p - 1) * tau  # (fill + steady)

        # Data-parallel replicas run in parallel, so latency is unchanged by d.
        # They DO reduce the number of steps needed to cover the whole dataset.

        cur_comp_time       = step_time * num_rounds  # wall-clock seconds
        cur_data_transfer   = main_root.calc_all_data_transfer(d, t, p)
        cur_comm_time       = main_root.all_comm_time(d, t, p, beta, num_mirco_batches_per_batch, batch_size) * num_rounds

        #mul by numer of batches in rounds

        cur_comm_time = main_root.all_comm_time(d, t, p, beta, num_mirco_batches_per_batch,
                                                batch_size) * num_rounds

        if (d, p, t) == show_config_tree:
            visualization.visualize_tree_dot(main_root, *show_config_tree)

        print(f"cfg={[d, t, p]}  transfer={cur_data_transfer}  comm={cur_comm_time:.5f}s  compute={cur_comp_time:.5f}s")
        results.append({
            "config":      [d, t, p],
            "transfer":    cur_data_transfer,
            "comm_time":   cur_comm_time,
            "compute_time":cur_comp_time,
            "flops":       main_root.calc_all_compute(),
        })

    # ------------------ WRITE OUT JSON ------------------
    out_dir  = Path(project_root) / "frontend" / "public"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "data.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"✅ wrote {len(results)} records to {out_file}")
# ============================ FUNCTIONS  ============================ #


def get_all_parallel_configs(total_gpu, min_t, max_p)-> list:
    all_configs = []
    for d_iter in range(1,total_gpu+1):
        for t_iter in range(min_t,int(total_gpu//d_iter)+1) :
            for p_iter in range(1,min(max_p+1, int(total_gpu//d_iter//t_iter)+1)):
                if [d_iter,t_iter,p_iter] not in all_configs and d_iter*t_iter*p_iter == total_gpu:
                    all_configs.append([d_iter,t_iter,p_iter])

    #print(f"All parallel configs: {all_configs}")
    return all_configs
def clear_nodes_names(nodes_list):
    for node in nodes_list:
        if "/model/" in node.name:
            node.name = node.name.replace("/model/", "")
    return nodes_list
def clear_nodes_list_redundancy(nodes_list: List[Node]) -> List[Node]:
    """
    1) Prune any “dead” nodes:
       - A node is “dead” if any of id_d, id_t, id_p, or layer < 0, or if "layer" not in its name.
       - Remove such nodes (and unlink them from parents/children).
    2) Merge all “mlp” chains into one node per layer:
       - For every node whose name contains "mlp", collect its non-mlp parents into mlp_parents[layer]
         and its non-mlp children into mlp_children[layer].
       - After collecting, remove only those “mlp” nodes (not the entire list).
       - Finally, for each layer i where mlp_parents[i] is nonempty, create a single Node("MLP_i")
         of op_type="compute" at layer=i, copy id_d/id_t/id_p from mlp_parents[i][0], then
         reattach all mlp_parents[i] → MLP_i → mlp_children[i], and append MLP_i to nodes_list.
    """

    # --- STEP 1: Prune “dead” nodes ---
    for node in nodes_list[:]:
        # Remove dead parents
        for p in node.parents[:]:
            if (
                p.id_d < 0 or p.id_t < 0 or p.id_p < 0
                or node.layer < 0
                or "layer" not in p.name
            ):
                node.remove_parent_child_link(p)

        # Remove dead children
        for c in node.children[:]:
            if (
                c.id_d < 0 or c.id_t < 0 or c.id_p < 0
                or node.layer < 0
                or "layer" not in c.name
            ):
                node.remove_parent_child_link(c)

        # If this node itself is “dead,” remove it
        if (
            node.id_d < 0
            or node.id_t < 0
            or node.id_p < 0
            or node.layer < 0
            or "layer" not in node.name
        ):
            nodes_list.remove(node)
            continue

        # Clear SkipLayerNorm connections (unchanged)
        if "SkipLayerNorm" in node.name:
            for c in node.children[:]:
                if "SkipLayerNorm" in c.name:
                    node.remove_parent_child_link(c)
            for p in node.parents[:]:
                if "SkipLayerNorm" in p.name:
                    node.remove_parent_child_link(p)

    # If, after pruning, the list is empty, return it immediately
    if not nodes_list:
        return nodes_list

    # --- STEP 2: Merge “mlp” chains by layer ---
    # 2a) Determine max layer to size our lists
    L_max = max(node.layer for node in nodes_list)
    mlp_parents = [[] for _ in range(L_max + 1)]
    mlp_children = [[] for _ in range(L_max + 1)]
    mlp_nodes = []

    # 2b) Collect all “mlp” nodes and record their non-mlp parents/children
    for node in nodes_list[:]:
        if "mlp" in node.name.lower():
            if node not in mlp_nodes:
                mlp_nodes.append(node)
                L = node.layer
                for p in node.parents:
                    if "mlp" not in p.name.lower():
                        if p not in mlp_parents[L]:
                            mlp_parents[L].append(p)
                for c in node.children:
                    if "mlp" not in c.name.lower():
                        if c not in mlp_children[L]:
                            mlp_children[L].append(c)

    # If no mlp nodes were found, return unchanged
    if not mlp_nodes:
        return nodes_list

    # 2c) Remove only those mlp_nodes from nodes_list (and unlink them)
    for m in mlp_nodes:
        if m in nodes_list:
            for p in list(m.parents):
                m.remove_parent_child_link(p)
            for c in list(m.children):
                m.remove_parent_child_link(c)
        nodes_list.remove(m)

    # 2d) For each layer i that had mlp_parents, create exactly one MLP_i node
    for i in range(len(mlp_parents)):
        if not mlp_parents[i]:
            continue  # no “mlp” chain in this layer

        # Use the first parent in mlp_parents[i] to copy id_d, id_t, id_p
        sample_parent = mlp_parents[i][0]
        MLP_node = Node(
            name= f"layer.{i}/MLP" ,
            op_type="MLP",
            layer=i
        )

        MLP_node.id_d = sample_parent.id_d
        MLP_node.id_t = sample_parent.id_t
        MLP_node.id_p = sample_parent.id_p
        MLP_node.node_type = "compute"

        # Attach all recorded parents → MLP_node
        for parent in mlp_parents[i]:
            parent.children = []
            parent.add_parent_child_link(MLP_node)

        # Attach MLP_node → all recorded children
        for child in mlp_children[i]:
            child.parents = []
            MLP_node.add_parent_child_link(child)

        # Finally, append MLP_node to nodes_list
        nodes_list.append(MLP_node)

    return nodes_list
def get_comm_nodes(root: Node) -> List[Node]:
    """
    Return a list of all communication nodes (node_type 'p2p' or 'collective')
    reachable from the given root. Also prints each unique op_type and its count.
    """
    visited = set()
    comm_nodes: List[Node] = []

    def dfs(node: Node):
        if node.index in visited:
            return
        visited.add(node.index)

        # Check if this node is a communication node (P2P or Collective)
        node_type_lower = (node.node_type or "").lower()
        if node_type_lower in ("p2p", "collective"):
            comm_nodes.append(node)

        for child in node.children:
            dfs(child)

    # Perform DFS from root
    dfs(root)

    # Count unique op_types among the communication nodes
    op_type_counts: Dict[str, int] = {}
    for n in comm_nodes:
        key = n.op_type or "UNKNOWN"
        op_type_counts[key] = op_type_counts.get(key, 0) + 1

    # Print each unique op_type and its count
    #print("Communication op_type counts:")
    for op_type, count in op_type_counts.items():
        print(f"  {op_type}: {count}", end=', ')

    return comm_nodes


if __name__ == "__main__":
    main()


