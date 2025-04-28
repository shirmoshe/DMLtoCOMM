import onnx_analyze
import class_Node
import result_visualization
from class_Node import Node
from collections import defaultdict
from class_GPU import GPU


def create_pipeline_stages(layers_dict, p):
    """
    Split model layers into p pipeline stages.

    Args:
        layers_dict (dict): {layer_index: list of Node}.
        p (int): Number of pipeline stages.

    Returns:
        dict: {stage_index: {layer_index: list of Node}}
    """

    # 1. Remove layer -1 (constant nodes etc.)
    layers_dict = {k: v for k, v in layers_dict.items() if k != -1}

    sorted_layers = sorted(layers_dict.items())
    total_layers = len(sorted_layers)
    layers_per_stage = total_layers // p
    extra_layers = total_layers % p

    # 3. Assign layers to stages
    stages = {}
    layer_idx = 0
    for stage_id in range(p):
        stage_layers = {}

        num_layers_in_stage = layers_per_stage + (1 if stage_id < extra_layers else 0)  # add extra layer
        for _ in range(num_layers_in_stage):
            if layer_idx < total_layers:
                layer_num, nodes = sorted_layers[layer_idx]
                stage_layers[layer_num] = nodes
                layer_idx += 1

        stages[stage_id] = stage_layers

    print("\n**********************************\nPipeline Stage Split:")
    for stage_id, layer_dict in stages.items():
        layers_in_stage = list(layer_dict.keys())
        print(f"Stage {stage_id}: Layers {layers_in_stage}")

    return stages


def create_send_recv_group(d_id, source_stage, dest_stage, t):
    """
    Create a list of GPU objects representing communication between two stages.

    Args:
        d_id (int): Data parallel index (fixed for this operation).
        source_stage (int): Source pipeline stage index.
        dest_stage (int): Destination pipeline stage index.
        t (int): Tensor parallel size.

    Returns:
        List of tuples: (source_gpu, destination_gpu)
    """
    connections = []

    for tensor_idx in range(t):
        src_gpu = GPU(source_stage, tensor_idx, d_id)
        dst_gpu = GPU(dest_stage,  tensor_idx, d_id)
        connections.append((src_gpu, dst_gpu))

    return connections


def create_pipeline_parallel(d, p, t, layers):
    for d_idx in range(d):
        stage_mapping = create_pipeline_stages(layers, p)
        result_visualization.create_stage_graph(stage_mapping, d_id=d_idx, output_dir="svg_file")

        for stage_id, stage_layers in stage_mapping.items():
            result_visualization.create_layer_graph(stage_layers, stage_id=stage_id, d_id=d_idx, output_dir="svg_file")

            connections = create_send_recv_group(d_idx, source_stage=stage_id, dest_stage=stage_id+1, t=t)
            result_visualization.create_send_recv_gpu_graph(connections, source_stage=stage_id, dest_stage=stage_id+1, d_id = d_idx)
