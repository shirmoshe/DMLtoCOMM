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
import pipeline_parallel


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



    # check parameters
    from collections import defaultdict

    def print_ops_by_layer(nodes_list):
        layer_ops = defaultdict(list)
        for node in nodes_list:
            layer = node.layer
            layer_ops[layer].append(node.op_type)

        for layer in sorted(layer_ops.keys()):
            print(f"\nlayer {layer}:")
            for op in layer_ops[layer]:
                print(f"  - {op}")


    # Create topology
  #  topology = Topology(topology_type, total_gpu)
  #  topology.add_GPU(d, t, p)

    # Create Node objects and build the hierarchy
    nodes_list = onnx_analyze.create_nodes(onnx_model)
    layers = onnx_analyze.group_layer(nodes_list)
    print_ops_by_layer(nodes_list=nodes_list)
    # ============================ DATA PARALLELISM ============================ #
    data_parllel.create_data_parallel(nodes_list, d, data_size)

    # ============================ PIPELINE PARALLELISM ============================ #
    pipeline_parallel.create_pipeline_parallel(d, p, t, layers)


if __name__ == "__main__":
    main()


