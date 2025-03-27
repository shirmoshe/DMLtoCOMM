import onnx
import onnx_analyze
import class_Node
import torch
# from onnx2pytorch import ConvertModel
import data_parllel
import os


def main():
    # Load the ONNX model
    model_path = "C:/Users/shirm/PycharmProjects/Project/imagenet/resnet50-v1-7.onnx"
    #model_path = "C:/Users/nadav/PycharmProjects/DMLtoCOMM/resnet50-v1-7.onnx"  # Replace with model path
    onnx_model = onnx_analyze.load_model(model_path)

    os.makedirs("svg_file", exist_ok=True)

    # Create Node objects and build the hierarchy
    nodes_list = onnx_analyze.create_nodes(onnx_model)

    # Generate an SVG graph using Graphviz
   # onnx_analyze.create_svg_graph(nodes_list, "my_onnx_graph")

    # data parallel
    d = 3

    model_replicas = data_parllel.create_data_parallel_collectives(nodes_list, d)

 #   merge_node_list = data_parllel.flatten_and_dedup(node_replicas)
 #   onnx_analyze.create_svg_graph(merge_node_list)
#  onnx_analyze.create_svg_graph_with_clusters(merge_node_list)

    onnx_analyze.create_interactive_high_level_svg(model_replicas)


    # Then generate each detailed replica view:
    for i, replica in enumerate(model_replicas):
        onnx_analyze.create_svg_graph(replica, output_file=f"gpu_{i}_detail")

if __name__ == "__main__":
    main()



