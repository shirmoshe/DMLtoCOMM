import onnx
import onnx_analyze
import class_Node
import torch
# from onnx2pytorch import ConvertModel
import data_parllel
import os
import json


def main():

    # ============================ INITIALIZATION ============================ #
    model_path = "imagenet/resnet50-v1-7.onnx"  # replace with model path
    onnx_model = onnx_analyze.load_model(model_path)  # load the ONNX model

    os.makedirs("svg_file", exist_ok=True)     # make folder for the svg file

    json_path = "user_inputs.json"      # load user inputs json file
    with open(json_path, 'r') as f:
        config = json.load(f)

    # extract parallelism parameters
    d = config['parallelism']['data_parallel_size']

    nodes_list = onnx_analyze.create_nodes(onnx_model)     # Create Node objects and build the hierarchy

    # ============================ DATA PARALLELISM ============================ #
    model_replicas = data_parllel.create_data_parallel_collectives(nodes_list, d)  # replica the model d times
    onnx_analyze.create_interactive_high_level_svg(model_replicas)      # create interactive high level graph
    for i, replica in enumerate(model_replicas):     # generate each detailed replica view
        onnx_analyze.create_svg_graph(replica, output_file=f"gpu_{i}_detail")


if __name__ == "__main__":
    main()



