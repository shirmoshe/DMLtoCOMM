from __future__ import annotations

import math


class Node:

    # Global counter for automatic unique IDs
    _id_counter: int = 10000

    def __init__(self, name, op_type, layer=0, index=None):
        self.name = name  # Original ONNX node name
        self.op_type = op_type  # Operation type (e.g., MatMul, Add)
        self.layer = layer

        self.index = index
        if index is None:
            self.index = Node._id_counter
            Node._id_counter += 1

        self.parents = []
        self.children = []
        self.gpu = None

        # whether this node does computation vs collective communication
        self.node_type = "compute"

        # shapes, sharding, and comm attributes
        self.shape_in = None
        self.shape_out = None
        self.num_reps_inside = 1
        self.data_transfer = 0
        self.comm_time = 0

        self.compute_ops = 0

        # other metadata
        self.data_size = 0
        self.id_d = 0
        self.id_t = 0
        self.id_p = 0
        self.is_tp_candidate = False
        self.shard_config = "COL"

    def __repr__(self):
        return (f"-------------------------------------------------------------------- \n"
                f"Node(index={self.index}, name={self.name!r}, layer={self.layer} \n "
                f"parents={[(p.index, p.name, p.id_d, p.id_t, p.id_p) for p in self.parents]}\n, "
                f"children={[(c.index, c.name, c.id_d, c.id_t, c.id_p) for c in self.children]})\n")

        #f"node_type={self.node_type}, op_type={self.op_type}, layer={self.layer}\n "
        #       f"GPU={gpu_info}, id_d={self.id_d}, id_t={self.id_t}, id_p={self.id_p}\n "

    def add_child(self, child: Node) -> None:
        """
        Create a directed edge from self → child.
        """
        if child not in self.children:
            self.children.append(child)


    def add_parent(self, parent: Node) -> None:
        """
        Create a directed edge parent → self.
        """
        if parent not in self.parents:
            self.parents.append(parent)

    def add_parent_child_link(self, child: Node):
        self.add_child(child)
        child.add_parent(self)

    def remove_child(self, child: Node) -> None:
        """
        Remove the directed edge self → child.
        """
        if child in self.children:
            self.children.remove(child)

    def remove_parent(self, parent: Node) -> None:
        """
        Remove the directed edge parent → self.
        """
        if parent in self.parents:
            self.parents.remove(parent)

    def remove_parent_child_link(self, child: Node):
        if child in self.children:
            self.children.remove(child)
        if self in child.parents:
            child.parents.remove(self)

    def clone(self):
        """
        Create a deep clone of this Node, copying all relevant attributes.

        Returns:
            Node: A new Node instance with identical fields.
        """
        new_node = Node(
            index= None,
            name=self.name,
            op_type=self.op_type,
            layer=self.layer
        )

        # Copy list of parents references
        new_node.parents = self.parents.copy()
        # Copy simple attributes
        new_node.gpu = self.gpu
        new_node.node_type = self.node_type
        new_node.data_size = self.data_size
        new_node.id_d = self.id_d
        new_node.id_t = self.id_t
        new_node.id_p = self.id_p
        new_node.is_tp_candidate = self.is_tp_candidate
        return new_node

    @staticmethod
    def print_nodes_list(nodes_list):
        """
        Print a list of Node objects.

        Args:
            nodes_list (list[Node]): List of nodes to print.
            use_repr (bool): If True, print __repr__, else print __str__.
        """
        for node in nodes_list:
                print(str(node))


    # ---------------------------------------------------------------
    # 1) Function to extract any tensor’s shape (e.g. (B, L, D)) from ONNX
    # ---------------------------------------------------------------

    # def _dims_from_value_info(value_info) -> tuple[int, ...] | None:
    #     """
    #     Helper: Given a ValueInfoProto, extract dims as ints.
    #     Return None if any dimension is symbolic or unknown.
    #     """
    #     dims = []
    #     for d in value_info.type.tensor_type.shape.dim:
    #         if d.dim_value > 0:
    #             dims.append(d.dim_value)
    #         else:
    #             # dim_value == 0 or dim_param means unknown
    #             return None
    #     return tuple(dims)

    def set_node_shape_out(self, d, t, p) -> None:
        # Default passthrough

        B_in, L_in, D_in = self.shape_in

        # Handle compute ops by op_type first
        if self.name.endswith(("LayerNorm", "SkipLayerNorm", "MLP")):
            # no shape change
            self.shape_out = [B_in, L_in, D_in]
            self.num_reps_inside = 1

        elif self.name.endswith("MatMul_Q4"):
            # split feature dimension across tensor-parallel
            self.shape_out = [B_in, L_in, D_in // t]
            self.num_reps_inside = t

        elif self.name.endswith("GroupQueryAttention"):
            # split feature dimension across tensor-parallel
            self.shape_out = [B_in, L_in, D_in]
            self.num_reps_inside = t

        # Then communications
        elif self.op_type == "READ_DATA_DP":
            # split batch across data-parallel
            self.shape_out = [int(B_in / d), L_in, D_in]
            self.num_reps_inside = 1

        elif self.op_type == "AllGather_TP":
            # reassemble tensor-parallel shards
            self.shape_out = [B_in, L_in, D_in * t]
            self.num_reps_inside = 1

        elif self.op_type == "AllReduce_TP":
            # reduce grads (shape unchanged)
            self.shape_out = [B_in, L_in, D_in]
            self.num_reps_inside = 1

        elif self.op_type == "AllReduce_DP":
            # reassemble data-parallel replicas
            self.shape_out = [B_in * d, L_in, D_in]
            self.num_reps_inside = 1

        # Fallback passthrough of input shape
        elif self.shape_out is None and self.shape_in:
            self.shape_out = list(self.shape_in)
            self.num_reps_inside = 1

        #print(f"# {self.name:<50}: {self.shape_out} * {self.num_reps_inside}")

        #print(f"[DEBUG] d={d}, t={t}, p={p} --- {self.op_type}:  shape_in: {self.shape_in}  ->  shape_out: {self.shape_out},reps: {self.num_reps_inside}")


    def rec_data_shape_flow(self, d, t, p):
        # Base case: stop if no children
        if not self.children:
            if self.op_type == "AllReduce_DP" and self.shape_out is None :
                self.set_node_shape_out(d, t, p)
            return

        # Compute this node’s output shape
        self.set_node_shape_out(d, t, p)

        # Propagate to children and recurse
        for child in self.children:
            # copy so parent’s list isn’t aliased
            child.shape_in = list(self.shape_out)
            child.rec_data_shape_flow(d, t, p)

    def calc_self_data_transfer(self, d, t, p): #according to ring

        if self.op_type == "READ_DATA_DP":
            self.data_transfer = math.prod(self.shape_out) * d * t * p

        elif self.op_type == "P2P":
            self.data_transfer = math.prod(self.shape_in) * t

        elif self.op_type == "AllGather_TP":
            # Only involves GPUs in tensor parallel group (size t)
            self.data_transfer = math.prod(self.shape_in) * (t - 1) * t * 3 # 3 is for Q K V

        elif self.op_type == "AllReduce_TP":
            # Only involves GPUs in tensor parallel group (size t)
            self.data_transfer = (2 * math.prod(self.shape_in)) * (t - 1) * t

        elif self.op_type == "AllReduce_DP":
            # Only involves GPUs in data parallel group (size d)
            self.data_transfer = (2 * math.prod(self.shape_in) * 4) * (d - 1) * d  # *4 is for W_q W_k W_v W_o
        else:
            self.data_transfer = 0

    def calc_all_data_transfer(self, d, t, p) -> float:
        visited = set()

        sum = self.data_transfer  # Start with current node's data transfer

        for child in self.children:
            if child.index not in visited:
                visited.add(child.index)
                child.calc_self_data_transfer(d, t, p)
                sum += child.calc_all_data_transfer(d, t, p)  # Add child's total
                #print(f"# {child.name:<50}: {child.shape_out} * {child.num_reps_inside}, data transfer: {child.data_transfer}")

        return sum

    def calc_self_comm_time(self, d, t, p, beta):
        # beta = 1/bandwidth

        # if self.op_type == "READ_DATA_DP":
        #     # Each GPU reads independently, no parallelization benefit
        #     self.comm_time = math.prod(self.shape_out) * beta

        if self.op_type == "P2P":
            # Point-to-point between pipeline stages, sequential
            self.comm_time = math.prod(self.shape_in) * beta

        elif self.op_type == "AllGather_TP":
            # t GPUs communicate in parallel within tensor parallel group
            # Each GPU receives from (t-1) others, but can receive in parallel
            self.comm_time = math.prod(self.shape_in) * (t - 1) * beta * 3 # 3 is for Q K V

        elif self.op_type == "AllReduce_TP":
            # Ring AllReduce within tensor parallel group
            # Communication happens in parallel across t GPUs
            self.comm_time = ((2 * math.prod(self.shape_in)) * (t - 1) * t / t) * beta

        elif self.op_type == "AllReduce_DP":
            # Global AllReduce across all GPUs
            # Communication can be parallelized across all participating GPUs
            total_gpus = d * t * p
            self.comm_time = ((2 * math.prod(self.shape_in)) * (d - 1) * d /d) * beta *4 # *4 is for W_q W_k W_v W_o


        else:
            self.comm_time = 0.0

        if self.op_type != "READ_DATA_DP" and self.op_type != "AllReduce_DP":
            self.comm_time = self.comm_time / p

    def calc_all_times(self, d, t, p, beta, num_mini_batches) -> float:
        visited = set()

        self.calc_self_comm_time(d, t, p, beta)
        sum = self.comm_time

        for child in self.children:
            if child.index not in visited:
                visited.add(child.index)
                sum += child.calc_all_times(d, t, p, beta , num_mini_batches)  # Child will calc its own time

        return sum

    def calc_self_compute(self) -> None:
        """
        Estimate FLOPs for compute-type nodes.

        Shape convention
        ----------------
        • Every tensor is assumed to have shape  [B, L, D]:
            B : Batch size         – number of independent sequences
                                     processed together in one forward / backward pass.
            L : Sequence length    – number of tokens (sub-words) in **each** sequence.
            D : Hidden size        – embedding / model dimension for **each** token.
        • Total number of tokens in the batch:
            M = B * L
        • Unless stated otherwise, a multiply-add counts as two FLOPs.
        • All comments below are in English only (per project guideline).
        """
        # Default: zero
        self.compute_ops = 0

        # Early exit for non-compute nodes
        if self.node_type != "compute":
            return

        # ----- Common dimensions -------------------------------------------------
        B, L, D = self.shape_in  # batch, sequence-length, hidden
        M = B * L  # total tokens in the whole batch

        # ------------------------------------------------------------------
        # MatMul family
        # ------------------------------------------------------------------
        if "MatMul" in self.op_type:
            # Shapes: (M × K) · (K × N)  →  (M × N)
            K = self.shape_in[2]
            N = self.shape_out[2]
            self.compute_ops = 2 * M * K * N

        # ------------------------------------------------------------------
        # LayerNorm variants
        # ------------------------------------------------------------------
        elif self.name.endswith(("LayerNorm", "SkipLayerNorm")):
            elems = math.prod(self.shape_out)
            self.compute_ops = 6 * elems  # ~6 FLOPs per element

        # ------------------------------------------------------------------
        # Two-layer MLP (Dense → activation → Dense)
        # ------------------------------------------------------------------
        elif self.op_type == "MLP":
            self.compute_ops = 16 * M * D * D  # 2*(M·D·4D) + 2*(M·4D·D)

        # ------------------------------------------------------------------
        # GroupQueryAttention
        # ------------------------------------------------------------------
        elif self.op_type == "GroupQueryAttention":   # batch, sequence-length, hidden
            attn_qk = 2 * B * L * L * D  # QKᵀ
            attn_softmax = 6 * B * L * L  # scale + softmax
            attn_v = 2 * B * L * L * D  # Attn·V
            total_attn = attn_qk + attn_softmax + attn_v

            # four linear projections (Q, K, V, O)
            proj_qkvo = 4 * 2 * M * D * D
            self.compute_ops = total_attn + proj_qkvo

        # ------------------------------------------------------------------
        # Fallback
        # ------------------------------------------------------------------
        else:
            self.compute_ops = math.prod(self.shape_out)

    def calc_all_compute(self) -> float:
        """
        Recursively sum compute_ops for this node and all downstream children,
        without double-counting any child (uses node.index to guard).
        """
        visited = set()
        total = self.compute_ops

        for child in self.children:
            if child.index not in visited:
                visited.add(child.index)
                # ensure child's compute_ops is up to date
                child.calc_self_compute()
                total += child.calc_all_compute()

        return total

    def calc_all_compute_time(self, flop_rate_per_gpu: float, t: int) -> float:
        """Return total compute-time (seconds) for this node + descendants."""
        # ensure *this* node has compute_time
        self.calc_self_compute_time(flop_rate_per_gpu, t)

        visited = set()
        total_time = self.compute_time

        for child in self.children:
            if child.index in visited:
                continue
            visited.add(child.index)
            child.calc_self_compute_time(flop_rate_per_gpu, t)
            total_time += child.calc_all_compute_time(flop_rate_per_gpu, t)

        return total_time

    def calc_self_compute_time(self, flop_rate_per_gpu: float, t: int):
        """
        Estimate this node’s compute time in seconds under tensor-parallelism:
          • flop_rate_per_gpu: FLOPs/sec each GPU can do
          • t: tensor-parallel degree (split FLOPs evenly across t GPUs)
        """
        # ensure compute_ops is set
        self.calc_self_compute()

        if self.node_type != "compute":
            self.compute_time = 0.0
        else:
            # perfect split across t GPUs
            self.compute_time = self.compute_ops / (flop_rate_per_gpu * t)

    def calc_stage1_compute_time(root_node, flop_rate_per_gpu: float, t: int, p: int) -> float:
        """
        Traverse the DAG from `root_node` and sum up each node’s compute_time
        (in seconds per micro-batch) for stage-1 nodes only.
        If p == 1 (no pipeline parallelism), all nodes are treated as stage 1.

        Parameters:
          root_node: Node          — entry point to your graph
          flop_rate_per_gpu: float — FLOP processing rate per GPU
          t: int                   — tensor-parallel degree (passed to calc_self)
          p: int                   — number of pipeline stages

        Returns:
          float — total compute time (seconds) per micro-batch for stage 1
        """
        visited = set()
        total_time = 0.0

        def dfs(node):
            nonlocal total_time
            if node.index in visited:
                return
            visited.add(node.index)

            # include this node if it's in stage 1, or if p==1 include all
            if p == 1 or node.id_p == 1:
                node.calc_self_compute_time(flop_rate_per_gpu, t)
                total_time += node.compute_time

            for child in node.children:
                dfs(child)

        dfs(root_node)
        return total_time

    # ------------- 1.  one-off cost for a single node -------------------------
    def single_comm_time(node,
                         d: int, t: int, p: int,
                         beta: float,
                         num_micro_batches: int) -> float:
        """
        Return comm-time (sec) of exactly ONE comm-node, regardless of stage.
        Useful when you want to profile a lone P2P edge.
        """
        node.calc_self_comm_time(d, t, p, beta, num_micro_batches)
        return node.comm_time

    # ------------- 2.  stage-1 aggregate (skip P2P by default) ---------------
    def calc_stage1_comm_time(root_node,
                              d: int, t: int, p: int,
                              beta: float,
                              num_micro_batches: int,
                              include_p2p: bool = False) -> float:
        """
        Return comm-time (sec) per micro-batch for stage-1 communication ops.
        By default excludes P2P edges (stage boundaries).  Set include_p2p=True
        to include them.
        """
        visited, total = set(), 0.0

        def dfs(node):
            nonlocal total
            if node.index in visited:
                return
            visited.add(node.index)

            in_stage1 = (p == 1) or (node.id_p == 1)
            is_p2p = (node.op_type == "P2P")

            if in_stage1 and (include_p2p or not is_p2p):
                node.calc_self_comm_time(d, t, p, beta)
                total += node.comm_time

            for ch in node.children:
                dfs(ch)

        dfs(root_node)
        return total

    def find_any_p2p_time(root_node,
                          d: int, t: int, p: int,
                          beta: float) -> float:
        """
        Return the comm-time (seconds) of the first P2P that sends data
        from pipeline stage-1 to stage-2 (smallest layer index).
        If no such P2P exists, return 0.0.
        """
        best_node = None
        best_layer = float("inf")

        stack, seen = [root_node], set()
        while stack:
            node = stack.pop()
            if node.index in seen:
                continue
            seen.add(node.index)

            if node.op_type == "P2P" and getattr(node, "p_src", None) == 1:
                if node.layer < best_layer:
                    best_layer = node.layer
                    best_node = node

            stack.extend(node.children)

        if best_node is None:
            #print("[DEBUG] find_any_p2p_time: no stage-1→2 P2P found")
            return 0.0

        best_node.calc_self_comm_time(d, t, p, beta)
        #print(f"[DEBUG] find_any_p2p_time: comm_time={best_node.comm_time}")
        return best_node.comm_time

    def find_AR_DP_time(root_node,
                          d: int, t: int, p: int,
                          beta: float) -> float:
        """
        Return the comm-time (sec) of the FIRST P2P node encountered on the
        left-most path of the DAG.  If none exists, return 0.
        """
        node = root_node
        while node and node.children:
            child = node.children[0]
            if child.op_type == "AllReduce_DP":
                child.calc_self_comm_time(d, t, p, beta)
                return child.comm_time
            node = child
        print("PROBLEM")
        return 0.0

    def all_comm_time(root_node,
                      d: int, t: int, p: int,
                      beta: float,
                      num_micro_batches_in_batch: int,
                      B: int) -> float:
        """

        Wall-clock comm-time for one forward-only step:
          • stage-1 comm  ×  (m + p – 1)
          • single P2P link × (m + p – 2)   [if such link exists]
        """
        stage1_comm_time = root_node.calc_stage1_comm_time(d, t, p, beta, num_micro_batches_in_batch)
        all_stages_time = stage1_comm_time * (num_micro_batches_in_batch + p - 1)

        p2p_one = root_node.find_any_p2p_time(d, t, p, beta)
        p2p_time = p2p_one * (num_micro_batches_in_batch + p - 2)

        ar_dp_time = root_node.find_AR_DP_time( d, t, p,beta)
        #ar_dp_time *= num_micro_batches_in_batch * B / d

        #print(f"TIMEs: tensor {all_stages_time:.5f}, pipeline {p2p_time:.5f}, data {ar_dp_time:.5f}")
        return all_stages_time + p2p_time + ar_dp_time

