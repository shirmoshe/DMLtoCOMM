from class_GPU import GPU
import networkx as nx
import matplotlib.pyplot as plt


class Topology:
    def __init__(self, topology_type, num_gpus, link_latency_ms=0):
        self.topology_type = topology_type  # e.g., "full_mesh", "fat_tree"
        self.num_gpus = num_gpus
        self.gpus = []           # List of GPU objects

    def add_GPU(self, d, t, p):
        gpu_list = []
        i = 0
        for dim_d in range(d):
            for dim_t in range(t):
                for dim_p in range(p):
                    gpu = GPU(i)
                    gpu.d_id = dim_d
                    gpu.t_id = dim_t
                    gpu.p_id = dim_p
                    gpu_list.append(gpu)
                    i += 1

        print("*************************************\n")
        for gpu in gpu_list:
            print("gpu id:", gpu.id, "    d:", gpu.d_id, "    t:", gpu.t_id, "    p:", gpu.p_id)

