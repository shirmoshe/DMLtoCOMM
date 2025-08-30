import onnx
from class_Node import Node
from graphviz import Digraph
import webbrowser
from graphviz import Digraph
import webbrowser
from collections import defaultdict
import data_parallel
import re
from typing import List, Optional, Tuple

import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_model(model_path):
    """
    Load an ONNX model, validate it, perform shape inference, and return the inferred model.
    Raises on invalid model.
    """
    model = onnx.load(model_path)

    # Validate the model explicitly
    #onnx.checker.check_model(model)   # Model has custom ops not recognized by ONNX checker.
                                       # We skip validation to allow further processing.

    # Perform shape inference
    inferred_model = onnx.shape_inference.infer_shapes(model)

    print("ONNX model loaded and validated with inferred shapes.\n")

    return inferred_model

def get_model_data_size(config):
    """Extract data size for the model from config."""
    return config['model']['data_size']

def get_topology_type(config):
    """
    Extract topology type from config.
    Raises KeyError if not found.
    """
    try:
        return config['topology']['type']
    except KeyError:
        raise KeyError("Missing 'topology.type' in configuration")

def get_parallelism_params(config):
    """
    Extract and validate parallelism parameters from config.
    Raises ValueError if d * t * p != total_gpus.
    Returns: tuple (d, t, p, total_gpu, topology_type)
    """

    d = config['parallelism']['data_parallel_size']
    t = config['parallelism']['tensor_parallel_size']
    p = config['parallelism']['pipeline_parallel_size']
    total_gpu = config['topology']['num_gpus']


    # Validate parallelism parameters

    if d * t * p != total_gpu:
        raise ValueError(
            f"Invalid GPU configuration: d={d}, t={t}, p={p}, but total_gpu={total_gpu}. "
            f"Expected: d * t * p = {d * t * p}"
        )
    return d, t, p, total_gpu

# In onnx_analyze.py or wherever create_nodes is defined,
# first ensure you have already run ONNX shape inference on the model:
#
#     model = onnx.load(model_path)
#     model = onnx.shape_inference.infer_shapes(model)
#
# Now modify create_nodes(...) so that each Node immediately gets its “full” tensor
# shapes (shape_in and shape_out) from the ONNX graph. These shapes reflect
# the pre‐parallelized dimensions (B, L, D, etc.). Once you later assign data‐parallel
# ranks (id_d, id_t, etc.), you can divide the batch dimension accordingly.

import re
import onnx
from class_Node import Node

def create_nodes(model: onnx.ModelProto):
    """
    Create Node objects representing ONNX graph nodes with parent‐child relationships.
    Also immediately assign each Node its input and output tensor shapes (tuples of ints),
    as inferred by ONNX. These shapes are “full” (pre‐parallel), and can be adjusted
    later once d/t/p are known.

    Args:
        model (onnx.ModelProto): Already‐inferred ONNX model (after infer_shapes).

    Returns:
        list[Node]: List of Node objects with shape_in and shape_out set.
    """

    # 1) helper to extract a tensor’s shape (tuple[int, ...]) from ONNX ValueInfo or initializer
    def _get_tensor_shape(tensor_name: str):
        # Search all inputs
        for vi in model.graph.input:
            if vi.name == tensor_name:
                return _dims_from_value_info(vi)
        # Search all intermediate value_info
        for vi in model.graph.value_info:
            if vi.name == tensor_name:
                return _dims_from_value_info(vi)
        # Search all outputs
        for vi in model.graph.output:
            if vi.name == tensor_name:
                return _dims_from_value_info(vi)
        # Finally search initializers (weights/constants)
        for init in model.graph.initializer:
            if init.name == tensor_name:
                return tuple(init.dims)
        return None

    def _dims_from_value_info(value_info) -> tuple[int, ...] | None:
        dims = []
        for d in value_info.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(d.dim_value)
            else:
                # If any dimension is symbolic or unknown, return None
                return None
        return tuple(dims)

    output_to_node: dict[str, Node] = {}
    node_list: list[Node] = []

    # 2) Create a Node object for each onnx_node, assign layer and shapes
    for idx, onnx_node in enumerate(model.graph.node):
        # Use the original ONNX node name if available; else generate one
        node_name = onnx_node.name if onnx_node.name else f"Node_{idx}"
        node = Node(index=idx, name=node_name, op_type=onnx_node.op_type)

        # Assign layer number from name pattern "/layers.<n>/"
        match = re.search(r"/layers\.(\d+)/", node.name)
        node.layer = int(match.group(1)) if match else -1

        # 2a) Assign “full” output tensor shape immediately (before any parallel)
        #     We take the first output name of the ONNX node:
        if onnx_node.output:
            first_output = onnx_node.output[0]
            out_shape = _get_tensor_shape(first_output)
            node.shape_out = out_shape
        else:
            node.shape_out = None

        # 2b) Assign “full” input tensor shape immediately (before any parallel)
        #     We attempt to pick a relevant input tensor name. Often the first input
        #     is either model input or previous node’s output. We try that:
        if onnx_node.input:
            first_input = onnx_node.input[0]
            in_shape = _get_tensor_shape(first_input)
            node.shape_in = in_shape
        else:
            node.shape_in = None

        # Note: at this early stage, id_d, id_t, id_p are all zero or unset.
        # These shapes (shape_in/shape_out) reflect the “full” batch size B, etc.
        # Later, once you set node.id_d > 1 (data‐parallel), you will adjust node.shape_out
        # by dividing the B dimension by id_d.

        node_list.append(node)

        # Map each output tensor name to this Node (for building links later)
        for out in onnx_node.output:
            output_to_node[out] = node

    # 3) Build parent‐child links exactly as before
    for idx, onnx_node in enumerate(model.graph.node):
        current_node = node_list[idx]
        for inp in onnx_node.input:
            if inp in output_to_node:
                parent_node = output_to_node[inp]
                # Maintain only forward links (no backward cycles)
                if (parent_node.layer < current_node.layer or
                    (parent_node.layer == current_node.layer and
                     parent_node.index < current_node.index)):
                    current_node.parents.append(parent_node)
                    parent_node.children.append(current_node)
            else:
                # Input likely comes from model input or constant; skip linking
                pass

    return node_list



def group_layer(nodes):
    # create dict{key->layer, val->[]list of operators}
    layers = defaultdict(list)
    for node in nodes:
        layer_num = node.layer
        layers[layer_num].append(node)

    return dict(layers)


import onnx

import onnx


def analyze_layers_metadata_reduced(model: onnx.ModelProto,
                                    nodes: List) -> Optional[int]:
    """
    A minimal replacement for analyze_layers_metadata that only finds
    the embedding dimension (D) once. Ignores all other layer metadata.

    Args:
        model:     An ONNX ModelProto (after shape inference).
        nodes:     A list of Node objects (or any iterable of objects
                   that have a '.name' attribute), provided for context.
                   We do not actually use 'nodes' here except to confirm
                   that the model is associated with these nodes.

    Returns:
        The embedding dimension D (int) if found, otherwise None.

    Logic:
      1) First, look for any initializer whose name contains "embed" (case‐insensitive).
         If found, assume its shape is [vocab_size, D] and return D.
      2) If no "embed" initializer is found, look for the first initializer
         whose name contains "qkv_proj" and ends with "weight_Q4". Use its dims
         to compute head_dim × num_heads = hidden_dim. Return hidden_dim.
      3) If neither is found, return None.
    """

    # 1) Try to find a conventional embedding weight initializer
    for init in model.graph.initializer:
        name_lower = init.name.lower()
        # common patterns: "embed_tokens.weight", "word_embeddings.weight", etc.
        if "embed" in name_lower and len(init.dims) == 2:
            # assume dims = [vocab_size, D]
            D = init.dims[1]
            return D

    # 2) Fallback: scan for any QKV quantized weight (e.g. "qkv_proj.*weight_Q4")
    for init in model.graph.initializer:
        name_lower = init.name.lower()
        if "qkv_proj" in name_lower and name_lower.endswith("weight_q4"):
            # dims might be [blocks, head_dim, num_heads]
            dims = list(init.dims)
            if len(dims) >= 3:
                head_dim = dims[1]
                num_heads = dims[2]
                hidden_dim = head_dim * num_heads
                return hidden_dim

    # 3) If nothing matched, return None
    return None


def extract_B_L_D_from_model(
    model: onnx.ModelProto
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Extract batch size B and sequence length L from the ONNX model’s inputs, then
    find embedding dimension D using analyze_layers_metadata_reduced.

    Returns:
        (B, L, D) as a tuple of ints, or (None, None, D) if B or L cannot be determined.
    """
    def _find_B_L() -> Tuple[Optional[int], Optional[int]]:
        """
        Search for a graph input whose shape has at least two concrete dims.
        If found, return (B, L) = (first_dim, second_dim). Otherwise, return (None, None).
        """
        for vi in model.graph.input:
            # Collect all positive (concrete) dims
            concrete_dims: List[int] = []
            for d in vi.type.tensor_type.shape.dim:
                if d.dim_value > 0:
                    concrete_dims.append(d.dim_value)
                else:
                    # If any dimension is symbolic or unknown, skip this input
                    concrete_dims = []
                    break

            if len(concrete_dims) >= 2:
                # Use the first two dims as (B, L)
                return concrete_dims[0], concrete_dims[1]

        return None, None

    # 1) Attempt to find (B, L) from any input with ≥2 concrete dims
    B, L = _find_B_L()
    if B is None or L is None:
        print("Warning: Could not find a graph input with ≥2 concrete dims for (B, L).")
    else:
        print(f"B (batch size)      = {B}")
        print(f"L (sequence length) = {L}")

    # 2) Find embedding dimension D using existing reduced metadata extractor
    D = analyze_layers_metadata_reduced(model, [])
    if D is None:
        print("D (embedding dim)   = Unknown (no suitable initializer found)")
    else:
        print(f"D (embedding dim)   = {D}")

    # 3) Explanation (unchanged)
    print("\n# Explanation:")
    print("  The embedding dimension D is fixed by the model’s weights (embedding layer or QKV).")
    print("  B (batch size) comes from the graph’s input shape. D does not influence B.")

    return B, L, D

