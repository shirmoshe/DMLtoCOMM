import onnx_analyze
import class_Node
import result_visualization
from class_Node import Node
from collections import defaultdict


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

    stages = {}

    # 3. Assign layers to stages
    for stage_id in range(p):
        stage_layers = {}

        num_layers_in_stage = layers_per_stage + (1 if stage_id < extra_layers else 0)  # add extra layer
        for layer_idx in range(num_layers_in_stage):
            if layer_idx < total_layers:
                layer_num, nodes = sorted_layers[layer_idx]
                stage_layers[layer_num] = nodes
                layer_idx += 1

        stages[stage_id] = stage_layers

    return stages



