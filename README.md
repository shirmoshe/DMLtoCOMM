# DML to COMM - Automated Distributed Machine Learning Communication Trace Generation
This project statically analyzes an ONNX model and a chosen parallelism configuration [d, t, p] (data, tensor, pipeline) to estimate communication volume, communication time, and compute time—without actual running a training.
It then writes the results to frontend/public/data.json, which a React app visualizes (tables + charts).

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
| File/Folder                   | Description                                                                     |
| :---------------------------- | :------------------------------------------------------------------------------ |
| `main.py`                     | Main entry point for model loading, parallelism configuration, and data export. |
| `onnx_analyze.py`             | Loads ONNX models and builds initial operator graphs.                           |
| `class_Node.py`               | Defines the `Node` class representing each operator.                            |
| `class_GPU.py`                | Defines the `GPU` class representing GPU topology (d, p, t coordinates).        |
| `data_parallel.py`            | Handles data parallel replication and collective operations.                    |
| `pipeline_parallel.py`        | Handles pipeline splitting and send/recv ops.                                   |
| `result_visualization.py`     | Creates SVG graphs for stages, layers, and GPU communication.                   |
| `svg_file/`                   | Output folder containing generated SVG files.                                   |
| `frontend/public/`            | React static assets (`index.html`, generated `data.json`).                      |
| `frontend/src/`               | React code (`App.tsx`, `index.tsx`, CSS, charts).                               |
| `frontend/tailwind.config.js` | Tailwind CSS configuration.                                                     |
| `frontend/postcss.config.js`  | PostCSS + Autoprefixer configuration.                                           |
| `frontend/package.json`       | Frontend dependencies and scripts.                                              |
| `load_model/code_files/user_inputs.json` | JSON configuration file controlling parallelism, topology, hardware, training, and visualization parameters. |
| `load_model/tiny_llama_model/tiny_llama.onnx` | Example ONNX model (Tiny-LLaMA) used for testing/analysis.                |


---

## ️ How It Works

**1. Model Parsing**

Load ONNX model.

Extract operators and build parent-child relationships.

**2. Parallelism Strategies**

**Data Parallelism:** Replicates the full model d times; adds Read and AllReduce ops.

**Pipeline Parallelism:** Splits layers into p stages; inserts Send/Recv ops.

**Tensor Parallelism:** Planned future work.

**3. Cost Estimation**

Communication costs estimated with simple bandwidth model (β = 1/bandwidth).

Compute costs estimated from operator FLOPs.

Graphviz generates SVGs for hierarchical task graphs.
React frontend loads data.json and displays config comparisons.

--- 
# Visualization -
**defultly diasabeld -to view go to `load_model/code_files/user_inputs.json`
and edit-
 {
  "visualization": {
    "show_config_tree": [0, 0, 0] **

to the config you want e.g. [6,2,2]
   - Interactive high-level graphs.
   - Detailed graphs for each data replica.
   - Stage-level graphs showing layer splits.
   - GPU graphs showing communication between stages.

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

## 🚀 How to Run the Project

**This project has two parts:**
 1. Backend (Python) – loads the ONNX model, applies the chosen parallelism configuration, and generates frontend/public/data.json
 2. Frontend (React) – visualizes the results (tables + charts) using data.json

**⚠️ Configuration (important!)**

Before running the backend, edit the JSON file:
   load_model/code_files/user_inputs.json
 This file defines:
   - parallelism (d / t / p sizes)
   - topology (GPUs, bandwidth, latency)
   - hardware (FLOP rate, etc.)
   - training (dataset size, batch size, micro-batches)
   - visualization (show_config_tree)

**➡️ To visualize a specific configuration tree, update "visualization" values inside user_inputs.json before running main.py**

---
# 🔧 Backend (Python) 

1. Create and activate a virtual environment (Windows example)
python -m venv .venv
.venv\Scripts\activate

2. Install Python dependencies
pip install -r requirements.txt

3. Run the analysis – this generates frontend/public/data.json
python main.py

# 🎨 Frontend (React)

1. Navigate to the frontend folder
cd frontend

2. Install project dependencies
npm install

3. (If not already configured) Install TailwindCSS and tools
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

4. Start the development server (http://localhost:3000)
npm.cmd start
---
