import onnx
import onnx_analyze
import class_Node
import torch
# from onnx2pytorch import ConvertModel
import data_parllel
import os
import json
from class_Topology import Topology
import result_visualization

def main():

    # ============================ INITIALIZATION ============================ #
    #model_path = "imagenet/resnet50-v1-7.onnx"  # replace with model path
    model_path = r"C:\Users\shirm\PycharmProjects\Project\load_model\tiny_llama_model\tiny_llama.onnx"  # replace with model path
    onnx_model = onnx_analyze.load_model(model_path)  # load the ONNX model

    os.makedirs("svg_file", exist_ok=True)     # make folder for the svg file

    json_path = r"C:\Users\shirm\PycharmProjects\Project\load_model\code_files\user_inputs.json"      # load user inputs json file
    with open(json_path, 'r') as f:
        config = json.load(f)

    data_size = config['model']['data_size']

    # ============================ TOPOLOGY ============================ #
    # extract parallelism parameters
    d = config['parallelism']['data_parallel_size']
    t = config['parallelism']['tensor_parallel_size']
    p = config['parallelism']['pipeline_parallel_size']
    total_gpu = config['topology']['num_gpus']
    topology_type = config['topology']['type']

    # Validate parallelism parameters
    if d * t * p != total_gpu:
        raise ValueError(f"Invalid GPU configuration: "
                         f"d={d}, t={t}, p={p}, but total_gpu={total_gpu}. "
                         f"Expected: d * t * p = {d * t * p}")

    # Create topology
    topology = Topology(topology_type, total_gpu)
    topology.add_GPU(d, t, p)

    # Create Node objects and build the hierarchy
    nodes_list = onnx_analyze.create_nodes(onnx_model)
    layers = onnx_analyze.group_layer(nodes_list)

    # ============================ DATA PARALLELISM ============================ #
    model_replicas = data_parllel.create_data_parallel_collectives(nodes_list, d, data_size)  # replica model d times
    result_visualization.create_interactive_high_level_svg(model_replicas)      # create interactive high level graph
    for i, replica in enumerate(model_replicas):     # generate each detailed replica view
        result_visualization.create_svg_graph(replica, output_file=f"data_{i}_detail")

    # ============================ PIPELINE PARALLELISM ============================ #

    for d_idx in range(d):
        result_visualization.create_layered_svg(layers, d_idx, "svg_file")


if __name__ == "__main__":
    main()



