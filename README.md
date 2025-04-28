DML to COMM
# Distributed Machine Learning Task Graph Generator

This project provides an automatic system for generating and visualizing task graphs for Distributed Machine Learning (DML), focusing on parallelism strategies: Data Parallelism, Pipeline Parallelism, and Tensor Parallelism.

---

##  Project Overview

The system:
- Parses an ONNX model into operator nodes.
- Splits the model according to different parallelism strategies.
- Visualizes the resulting computation and communication graphs in interactive SVG format.
- Supports hierarchical graph navigation:
  - Data replicas
  - Pipeline stages
  - Layers within stages
  - Operator-level details

---

##  Project Structure

| File/Folder | Description                                                                       |
|:------------|:----------------------------------------------------------------------------------|
| `main.py` | Main entry point for model loading, parallelism configuration, and visualization. |
| `onnx_analyze.py` | Loads ONNX models and builds initial operator graphs.                             |
| `class_Node.py` | Defines the `Node` class representing each operator.                              |
| `class_GPU.py` | Defines the `GPU` class representing GPU topology (d, p, t coordinates).          |
| `data_parllel.py` | Handles data parallel replication and collective operations.                      |
| `pipeline_parallel.py` | Handles pipeline splitting, send_recv ops.                                        |
| `result_visualization.py` | Creates SVG graphs for stages, layers, and GPU communication.                     |
| `svg_file/` | Output folder containing generated SVG files.                                     |

---

## ️ How It Works

1. **Model Parsing**  
   - Load ONNX model.
   - Extract operators and build parent-child relationships.

2. **Parallelism Strategies**  
   - **Data Parallelism**:  
     - Replicates the full model `d` times.
     - Adds Read and AllReduce collective ops.
   - **Pipeline Parallelism**:  
     - Splits model layers into `p` pipeline stages.
     - Inserts Send-Recv operations between stages.
     - Each stage contains a subset of layers.
   - **Tensor Parallelism** (future work).

3. **Visualization**  
   - Interactive high-level graphs.
   - Detailed graphs for each data replica.
   - Stage-level graphs showing layer splits.
   - GPU graphs showing communication between stages.

---

##  Visualization Examples

- **Data Parallel High-Level Graph**  
  Shows all data replicas and collective communication.

- **Pipeline Stage Graph**  
  Shows how layers are grouped into pipeline stages.

- **GPU Communication Graph**  
  Shows how GPUs at different stages exchange activations.

---

##  Example Navigation

1. Start from `data_0_detail.svg` (overview of a replica).
2. Click on a stage (e.g., Stage 0) → open `stage_0_data_0_detail.svg`.
3. Click on a layer (e.g., Layer 5) → open `layer_5_data_0_detail.svg`.
4. Click on a Send-Recv operation → open GPU communication graph.

---

##  Requirements

- Python 3.8+
- Packages:
  - `onnx`
  - `graphviz`
  - `torch`
  - `webbrowser`
  - `collections`

Graphviz must be installed on your system for SVG generation.

---

## Future Work

- Adding Tensor Parallelism split.
- More realistic modeling of microbatch flow.
- Memory usage and communication cost estimation.

---

## Author

---