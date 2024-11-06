# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
SPMD solver similar to intra_op of Alpa
"""
from typing import List, Dict, Optional, Tuple
import logging
import more_itertools
import multiprocessing
import numpy as np
import warnings
import time

from cube.ir.operator import IRFwOperation
from cube.graph.function.dimops import IRDimops, TransformRule, DimopSplit

from .block import IRBlock
from ..estimator.cost_model import CostModel
from ..plan.plan import StageSpec

# ILP solver
import pulp
from pulp import LpVariable, LpProblem, LpMinimize, LpStatus, lpSum, lpDot, LpStatus


_logger = logging.getLogger(__name__)


Nodes = Tuple[IRBlock]


class SpmdSolver:

    def __init__(self, cost_model: CostModel, recompute: bool,
                 memory_saving: bool = True):
        """SPMD Solver for searching the best spmd parallelism configuration.

        Note the operators that are assigned to devices will not be searched.

        Args:
            cost_model (CostModel): cost model for communication and computation
            recompute (bool): whether to apply recompute
            memory_saving (bool): True to remove replication of nodes if there are other
                partitioning choices for each node.
        """
        self.cost_model: CostModel = cost_model
        self.recompute: bool = recompute
        self.memory_saving: bool = memory_saving
        # device idx -> param memory limit
        self.device_mem_limit_bytes: Dict[int, int] = {}

        # (blocks, dp_size, tp_size) -> StageSpec
        self._cache: Dict[Tuple[Nodes, int, int], StageSpec] = {}

    def add_device_mem_limit(self, device: int, limit_gb: float):
        self.device_mem_limit_bytes[device] = int(limit_gb * 1024 * 1024 * 1024)

    def clear(self):
        self._cache = {}

    def solve(self, blocks: List[IRBlock],
              devices: Tuple[int],
              inflights: int,
              memory_limit: int,
              init_mem: int = 0,
              init_comp: float = 0.,
              min_dp: int = 1, max_dp: int = 32,
              min_tp: int = 1, max_tp: int = 32,) -> Optional[StageSpec]:

        device_constraints = set()
        _logger.debug(f'solve for blocks {blocks[0].bid}-{blocks[-1].bid}; devices: {devices}')
        for blk in blocks:
            if blk.devices is not None:
                device_constraints.add(blk.devices)
        # at least two blocks have conflicts on device constraints,
        # therefore cannot be in one stage
        if len(device_constraints) > 1:
            return None
        # The stage device doesn't satisfy constaints of blocks
        if len(device_constraints) == 1:
            if devices not in device_constraints:
                return None

        # setup device memory constraint
        for devid in devices:
            if devid in self.device_mem_limit_bytes:
                memory_limit = min(memory_limit, self.device_mem_limit_bytes[devid])
        
        min_dp = max(min_dp, max(blk.min_dp for blk in blocks))
        max_dp = min(max_dp, min(blk.max_dp for blk in blocks))
        min_tp = max(min_tp, max(blk.min_tp for blk in blocks))
        max_tp = min(max_tp, min(blk.max_tp for blk in blocks))

        if min_dp > max_dp or min_tp > max_tp:
            return None

        # True for 1, 2, 4, 8, 16,...
        is_of_power2 = lambda n: (n & (n-1) == 0) and n != 0

        best_stage_spec = None
        min_latency = None

        no_solution = False
        best_tp = -1
        for dp in range(min_dp, min(len(devices) + 1, max_dp + 1)):
            # constraints: only search for gpu# of power of 2
            if not is_of_power2(dp): continue
            # get tp size
            if len(devices) % dp != 0: continue
            tp = len(devices) // dp

            if not (min_tp <= tp <= max_tp): continue
            # constraints: only search for gpu# of power of 2
            if not is_of_power2(tp): continue

            # this means a larger tp size is already infeasible to
            # satisfy memory limit
            if no_solution:
                self._cache[self.get_key(blocks, dp, tp)] = None
                continue

            spec = self._solve(blocks, dp, tp, inflights, memory_limit,
                               init_mem, init_comp)
            # no solution -- the later smaller tp will also be infeasible
            if spec is None:
                no_solution = True
                continue

            if min_latency is None or spec.est_latency < min_latency:
                best_stage_spec = spec
                best_tp = tp
                min_latency = spec.est_latency
        # _logger.debug(f'best_tp: {best_tp}')
        return best_stage_spec

    def get_key(self, blocks: List[IRBlock], dp_size: int, tp_size: int):
        """Get the key of the solved problem"""
        return (tuple(blocks), dp_size, tp_size)
    
    def split_nodes_on_dp(self, nodes: List[IRFwOperation], dp_size: int):
        dp_size = 1
        """Split the nodes based on data parallelism size"""
        fnodes = []
        for node in nodes:
            algo = node.algorithms('dim')
            sub_node = algo.instantiate(idx=node.dp_idx, dim=node.dp_dim, num=dp_size)[0]
            sub_node.copy_args(node)
            # TODO: sub_node
            fnodes.append(node)
        return fnodes
    
    def get_stage_est_cost(self, fnodes: List[IRFwOperation], stage: StageSpec, inflights: int):
        
        total_span = 0
        splits = []
        cid2splits = {}
        factor = stage.dp_size
        for fnode in fnodes:
            split = stage.tp_spec[fnode.cid]
            # append data parallelism config
            # idxs.append(node.dp_idx if isinstance(node, IRDimops) else None)
            # dims.append(node.dp_dim if isinstance(node, IRDimops) else None)
            # nums.append(dp)
            # append tensor parallelism config
            if split is None:
                splits.append(None)
            else:
                idxs, dims, nums = [], [], []
                for i in range(0, len(split), 3):
                    idxs.append(split[i])
                    dims.append(split[i+1])
                    nums.append(split[i+2])
                splits.append([idxs, dims, nums])
            cid2splits[fnode.cid] = splits[-1]

        # computation cost
        span, mem_cost = self.cost_model.estimator(
                fnodes, splits, inflights) 
        mem_cost = mem_cost
        total_span += span / factor

        # communication cost
        reshard_cost = 0
        edges = self.cost_model.get_edges(fnodes)
        for fnode_src, fnode_dsts in edges.items():
            for fnode_dst in fnode_dsts:
                tensor_idxs = self.cost_model.get_vtensor_to_idxs(fnode_src, fnode_dst)
                for tensor, (idx_src, idx_dst) in tensor_idxs.items():
                    rule_src = None
                    if cid2splits[fnode_src.cid] is not None:
                        idxs, dims, num = cid2splits[fnode_src.cid]
                        rule_src = fnode_src.algorithms('dim').infer(idxs, dims, num)
                    rule_dst = None
                    if cid2splits[fnode_dst.cid] is not None:
                        idxs, dims, num = cid2splits[fnode_dst.cid]
                        rule_dst = fnode_dst.algorithms('dim').infer(idxs, dims, num)
                    reshard_cost += self.cost_model.comm_cost(
                            tensor, stage.tp_size, 
                            rule_src.outputs()[idx_src] if rule_src is not None else DimopSplit(r=True),
                            rule_dst.inputs()[idx_dst] if rule_dst is not None else DimopSplit(r=True),
                            stage.tp_spec[fnode_dst.cid] is None
                        )
        total_span += reshard_cost / factor
        total_span = total_span / 3 * 4 if self.recompute else total_span
        return total_span
            
    def get_partitions_from_partials(self, tp_size: int, partials: List[int]) -> List[int]:
        partitions = []
        for i in partials:
            partitions.append(i)
            tp_size = tp_size // i
        partitions.append(tp_size)
        return partitions

    def _solve(self,
               blocks: List[IRBlock],
               dp_size: int,
               tp_size: int,
               inflights: int,
               memory_limit: int,
               init_mem: int,
               init_comp: float) -> Optional[StageSpec]:
        """
        Search for the best spmd parallelism configuration given parallelism size.
        The search is only suitable for training.

        Args:
            blocks (List[IRBlock])
            dp_size (int): data parallel size
            tp_size (int): tensor parallel size
            inflights (int): maximal inflight micro-batches
            memory_limit (int): memory upper bound
            
        Returns:
            spec (StageSpec | None): operator transformation configuration
                None indicates no solution given by memory limit.
        """
        key = self.get_key(blocks, dp_size, tp_size)
        if key in self._cache:
            # print(f"cache hit: {key}")
            return self._cache[key]

        tic = time.time()

        fnodes: List[IRFwOperation] = list(more_itertools.flatten(blk.nodes for blk in blocks))
        fnodes = self.cost_model.get_search_nodes(fnodes)
        fnodes = self.split_nodes_on_dp(fnodes, dp_size)
        factor = dp_size
        # fnodes_with_anchor

        # create variables (nodes)
        s, d, c = {}, {}, {}  # partition index, computation cost, communication cost
        e, r = [], []  # inter-node resharding cost

        num_nodes = 0
        for fnode in fnodes:
            cid = fnode.cid
            algos = self.cost_model.partition_algos[fnode.cid]
            npartitions = len(algos)
            s[cid] = LpVariable.matrix(f's[{num_nodes}]', (range(npartitions),), cat='Binary')
            d[cid] = self.cost_model.get_comp_cost(fnode, tp_size).flatten() / factor
            c[cid] = self.cost_model.get_comm_cost(fnode, tp_size).flatten() / factor
            assert len(s[cid]) == len(d[cid]) == len(c[cid])
            # setup initial value
            for pidx, strategy in enumerate(algos):
                if strategy is None: continue
                idxs, dims, partials = strategy
                partitions = self.get_partitions_from_partials(tp_size, partials)
                for idx, dim, partition in zip(idxs, dims, partitions):
                    identifier = fnode.anno.input(idx)[dim].identifiers[0]
                    # we constrain a node that can only be evenly partitioned
                    if fnode.anno.getlen(identifier) % partition != 0:
                        s[cid][pidx].setInitialValue(False)
                        s[cid][pidx].fixValue()
                        npartitions -= 1
                        break
            # remove replicate choice if we have other choices to
            # partition nodes to save memory
            if self.memory_saving and npartitions > 1 and algos[0] is None:
                s[cid][0].setInitialValue(False)
                s[cid][0].fixValue()
                npartitions -= 1
            if npartitions <= 0:
                raise RuntimeError(
                    f"Infeasible problem: cannot find a partition choice for node: {fnode.name}[{cid}] "
                    f"in problem tp={tp_size}, dp={dp_size}")
            num_nodes += 1

        edges = self.cost_model.get_edges(fnodes)
        num_edges = 0
        for src, dsts in edges.items():
            for dst in dsts:
                nsrc = len(self.cost_model.partition_algos[src.cid])
                ndst = len(self.cost_model.partition_algos[dst.cid])
                e.append(LpVariable.matrix(f"e[{src.cid}, {dst.cid}]",
                                           (range(nsrc * ndst),),
                                           cat='Binary'))
                r.append(self.cost_model.get_pair_reshard_cost(src, dst, tp_size).flatten() / factor)
                num_edges += 1

        # initial value: --skip

        # objective
        prob = LpProblem('spmd', LpMinimize)
        # computation cost
        obj = 0
        for fnode in fnodes:
            cid = fnode.cid
            obj += lpDot(s[cid], d[cid]) # + lpDot(s[cid], c[cid])
        # communication cost
        for i in range(num_edges):
            obj += lpDot(e[i], r[i])

        prob += obj

        # constraints

        # a) only one partition can be selected
        for fnode in fnodes:
            prob += lpSum(s[fnode.cid]) == 1
        for i in range(num_edges):
            prob += lpSum(e[i]) == 1

        # e_src_dst[i][j] = 1 => s_src[i] == 1 and s_dst[j] == 1
        eidx = 0
        for src, dsts in edges.items():
            for dst in dsts:
                C = len(s[dst.cid]) if dst.cid in s else 1
                R = len(s[src.cid]) if src.cid in s else 1

                if src.cid in s:
                    for row in range(R):
                        prob += lpSum(
                            e[eidx][row * C + col] for col in range(0, C)) <= s[src.cid][row]
                if dst.cid in s:
                    for col in range(C):
                        prob += lpSum(
                            e[eidx][row * C + col] for row in range(0, R)) <= s[dst.cid][col]
                eidx += 1

        # b) memory constraint --skip

        assert "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True), (
            "Please install ILP solvers by 'sudo apt install coinor-cbc' or 'pip install pulp'")

        time_limit = 600
        solver = pulp.PULP_CBC_CMD(
            mip=True, msg=0, 
            timeLimit=time_limit, 
            threads=multiprocessing.cpu_count())
        prob.solve(solver)

        status = prob.status
        if status == pulp.LpStatusInfeasible:
            raise RuntimeError(
                f"infeasible problem: {len(blocks)} blocks, tp={tp_size}, dp={dp_size}. "
                f"Please report the bug")
        elif status == pulp.LpStatusUndefined:
            raise RuntimeError(
                f"Cannot find solution of the problem within time limit: "
                f"{len(blocks)} blocks, tp={tp_size}, dp={dp_size}"
            )

        objective = pulp.value(prob.objective)
        objective = float(objective) if objective is not None else -1.0
        # print(f"ILP Status: {LpStatus[status]}\tObjective: {objective}")
        # print(f"#nodes: {num_nodes},  #edges: {num_edges}")
        # print(f'ILP search time: {time.time() - tic:.2f} seconds')

        # reshard_cost = 0
        # for i in range(num_edges):
        #     reshard_cost += lpDot(e[i], r[i])
        # reshard_cost = pulp.value(reshard_cost)
        # print(f'debug info: reshard cost: {reshard_cost}')

        def get_non_zero_index(binary_vector):
            """Get the index of non-zero item in a vector."""
            ct = 0
            ret = None
            for i, elem in enumerate(binary_vector):
                if pulp.value(elem):
                    ret = i
                    ct += 1

            assert ct == 1
            return ret

        tp_spec: Dict[int, int] = {}
        for fnode in fnodes:
            index = get_non_zero_index(s[fnode.cid])
            tp_spec[fnode.cid] = index

        if objective > 1e13:
            warnings.warn("Detect unexpected behaviors in the auto-sharding pass.")

        # get tensor parallelism spec
        stage_tp_spec = {}
        names = {}
        for fnode in fnodes:
            split = None if tp_size == 1 else \
                self.cost_model.partition_algos[fnode.cid][tp_spec[fnode.cid]]
            stage_tp_spec[fnode.cid] = None if split is None else (
                split[0], split[1], self.get_partitions_from_partials(tp_size, split[2]))
            names[fnode.cid] = fnode.name

        # estimate memory
        splits = []
        for fnode in fnodes:
            split = stage_tp_spec[fnode.cid]
            splits.append(split)
        # assume adam optimizer
        span, mem_cost = self.cost_model.estimator(fnodes, splits, inflights)
        mem_cost += init_mem
        if mem_cost > memory_limit:
            mem_gb = round(mem_cost/1024/1024/1024, 2)
            _logger.debug(f'results of {len(tp_spec)} nodes: tp={tp_size}, dp={dp_size}: no solution (memory: {mem_gb} GB)')
            stage = None
        else:
            objective = objective + init_comp
            for fnode in fnodes:
                split = stage_tp_spec[fnode.cid]
                if split:
                    flatten_split = []
                    for i in range(len(split[0])):
                        flatten_split += [split[0][i], split[1][i], split[2][i]]
                    stage_tp_spec[fnode.cid] = flatten_split
            stage = StageSpec(
                est_latency=objective / 3 * 4 if self.recompute else objective,
                est_memory=mem_cost,
                tp_size=tp_size,
                dp_size=dp_size,
                tp_spec=stage_tp_spec,
                names=names,
            )
            est_cost = self.get_stage_est_cost(fnodes, stage, inflights)
            if abs(est_cost - stage.est_latency) > 1e-3:
                _logger.debug(f"Estimation error: {est_cost} vs {stage.est_latency}")
            _logger.debug(f'results of {len(stage_tp_spec)} nodes: tp={tp_size}, dp={dp_size} '
                          f'lat={round(stage.est_latency, 2)} ms, cost={round(est_cost, 2)} ms, '
                          f'mem={round(mem_cost/1024/1024/1024, 2)} GB')
        self._cache[key] = stage
        return stage
