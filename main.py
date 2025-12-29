import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import logging
from typing import List, Literal, Dict, Tuple
from pydantic import BaseModel, Field, model_validator

# --- 0. PERFORMANCE SETUP ---
# Attempt to load Numba for C-speed compilation. 
# If missing, simulation will run but slower.
try:
    from numba import jit
    JIT_AVAILABLE = True
    print("🚀 Numba JIT detected: High-Performance Mode ACTIVE.")
except ImportError:
    # Shim decorator if Numba is missing
    def jit(nopython=True):
        def decorator(func): return func
        return decorator
    JIT_AVAILABLE = False
    print("⚠️ Numba not found: Running in pure Python (Slow). Install 'numba' for speed.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("GridEngine_v9")

# ==========================================
# 1. STRICT DATA MODELS (The Contract)
# ==========================================
class Bus(BaseModel):
    id: str
    voltage_level_kv: float = Field(..., gt=0)
    type: Literal['PQ', 'PV', 'SLACK'] = 'PQ'

class ACLine(BaseModel):
    id: str
    from_bus: str
    to_bus: str
    length_km: float = Field(..., gt=0)
    r_ohm_per_km: float = Field(..., ge=0)
    x_ohm_per_km: float = Field(..., ge=0)
    
    @property
    def z_total(self) -> complex:
        return complex(self.r_ohm_per_km, self.x_ohm_per_km) * self.length_km

class Load(BaseModel):
    id: str
    bus: str
    p_mw: float
    q_mvar: float

class GridTopology(BaseModel):
    """
    Single Source of Truth. Validates connectivity before physics starts.
    """
    buses: List[Bus]
    lines: List[ACLine]
    loads: List[Load] = []

    @model_validator(mode='after')
    def validate_connectivity(self):
        bus_ids = {b.id for b in self.buses}
        for line in self.lines:
            if line.from_bus not in bus_ids or line.to_bus not in bus_ids:
                raise ValueError(f"❌ Orphaned line detected: {line.id} connects to unknown bus.")
        return self

# ==========================================
# 2. TOPOLOGY GENERATOR (Scalability Tool)
# ==========================================
def generate_linear_feeder(n_buses: int, voltage_kv: float = 20.0) -> Tuple[List[Bus], List[ACLine]]:
    """
    Programmatically generates a grid of size N.
    Topology: Source -> B1 -> B2 ... -> Bn
    """
    buses = [Bus(id="source", voltage_level_kv=voltage_kv, type='SLACK')]
    lines = []
    
    prev_id = "source"
    for i in range(1, n_buses + 1):
        curr_id = f"bus_{i}"
        
        # Add Bus
        buses.append(Bus(id=curr_id, voltage_level_kv=voltage_kv))
        
        # Add Line (Uniform impedance for simplicity)
        lines.append(ACLine(
            id=f"line_{i}", 
            from_bus=prev_id, 
            to_bus=curr_id, 
            length_km=1.0,  # 1 km segments
            r_ohm_per_km=0.15, 
            x_ohm_per_km=0.11
        ))
        prev_id = curr_id
        
    return buses, lines

# ==========================================
# 3. HIGH-PERFORMANCE PHYSICS KERNEL
# ==========================================
@jit(nopython=True)
def _fast_fbs_kernel(
    v_mag: np.ndarray, 
    v_complex: np.ndarray,
    p_inj: np.ndarray, 
    q_inj: np.ndarray, 
    z_matrix: np.ndarray, 
    parent_indices: np.ndarray,
    order_indices: np.ndarray,
    tol: float,
    max_iter: int
) -> Tuple[np.ndarray, bool]:
    """
    The mathematical core. Compiles to machine code via Numba.
    Performs Forward-Backward Sweep in microseconds.
    """
    n_nodes = len(v_complex)
    converged = False

    for _ in range(max_iter):
        max_err = 0.0
        
        # 1. Calc Current Injections (Safe Division)
        i_inj = (p_inj - 1j * q_inj) / (v_complex + 1e-9)
        
        # 2. Backward Sweep (Sum Currents from Leaves to Root)
        i_branch = np.zeros(n_nodes, dtype=np.complex128)
        for k in range(n_nodes - 1, 0, -1): # Reverse Order
            idx = order_indices[k]
            i_branch[idx] += i_inj[idx]
            parent = parent_indices[idx]
            if parent != -1:
                i_branch[parent] += i_branch[idx]

        # 3. Forward Sweep (Calc Voltages from Root to Leaves)
        for k in range(1, n_nodes): # Forward Order
            idx = order_indices[k]
            parent = parent_indices[idx]
            
            z = z_matrix[idx]
            v_new = v_complex[parent] - (i_branch[idx] * z)
            
            err = np.abs(v_new - v_complex[idx])
            if err > max_err: max_err = err
            
            v_complex[idx] = v_new
            v_mag[idx] = np.abs(v_new)

        if max_err < tol:
            converged = True
            break
            
    return v_complex, converged

class FastGridEngine:
    """
    Manages mapping between Pydantic Objects and Numpy Arrays.
    """
    def __init__(self, grid: GridTopology):
        self.grid = grid
        self._compile_topology()
    
    def _compile_topology(self):
        # 1. Index Mapping
        self.bus_map = {b.id: i for i, b in enumerate(self.grid.buses)}
        self.idx_map = {i: b.id for b.id, i in self.bus_map.items()}
        n = len(self.grid.buses)
        
        # 2. Graph Analysis (NetworkX)
        G = nx.Graph()
        for line in self.grid.lines:
            G.add_edge(line.from_bus, line.to_bus, z=line.z_total)
            
        root_id = next((b.id for b in self.grid.buses if b.type == 'SLACK'), self.grid.buses[0].id)
        
        # 3. Build Vectorized Structures
        bfs_tree = nx.bfs_tree(G, source=root_id)
        self.order = [self.bus_map[node] for node in bfs_tree]
        
        self.z_array = np.zeros(n, dtype=np.complex128)
        self.parents = np.full(n, -1, dtype=np.int32)
        self.base_kv = np.array([b.voltage_level_kv for b in self.grid.buses])
        
        for u, v in bfs_tree.edges():
            idx_u, idx_v = self.bus_map[u], self.bus_map[v]
            # u is parent in BFS tree
            self.parents[idx_v] = idx_u
            self.z_array[idx_v] = G.edges[u, v]['z']

    def solve(self, active_loads: List[Load]) -> Dict[str, complex]:
        n = len(self.grid.buses)
        
        # Map Objects -> Arrays
        p_inj = np.zeros(n)
        q_inj = np.zeros(n)
        for load in active_loads:
            if load.bus in self.bus_map:
                idx = self.bus_map[load.bus]
                p_inj[idx] += load.p_mw
                q_inj[idx] += load.q_mvar

        # Run Kernel
        v_complex = self.base_kv.astype(np.complex128)
        v_res, conv = _fast_fbs_kernel(
            np.abs(v_complex), v_complex, p_inj, q_inj, 
            self.z_array, self.parents, np.array(self.order),
            tol=1e-5, max_iter=20
        )
        
        if not conv: logger.warning("⚠️ Solver divergence detected")
        return {self.idx_map[i]: v_res[i] for i in range(n)}

# ==========================================
# 4. VECTORIZED CONTROLLER
# ==========================================
class VectorizedController:
    """
    Computes control actions for 1000s of buses in one SIMD instruction.
    """
    def compute_batch_q(self, v_complex_dict: Dict[str, complex], base_kv: np.ndarray) -> np.ndarray:
        v_vals = np.array(list(v_complex_dict.values()))
        v_pu = np.abs(v_vals) / base_kv
        
        err = 1.0 - v_pu
        q_out = np.zeros_like(err)
        
        # Logic: If Voltage > 1.05 p.u., Absorb Reactive Power (Linear Ramp)
        high_mask = (err < -0.05)
        q_out[high_mask] = np.clip((err[high_mask] + 0.05) * 20.0, -1.0, 0.0)
        
        return q_out

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n=== GRID ENGINE: SCALABILITY TEST ===\n")

    # --- A. GENERATE TOPOLOGY (The "Increase Bus Bar" fix) ---
    N_BUSES = 50  # <--- CHANGE THIS NUMBER to 10, 100, or 1000
    print(f"[*] Generating Grid with {N_BUSES} buses...")
    
    buses, lines = generate_linear_feeder(N_BUSES, voltage_kv=20.0)
    
    # Define Load at the END of the line (Worst Case Scenario)
    end_bus_id = buses[-1].id
    loads = [Load(id="heavy_load", bus=end_bus_id, p_mw=8.0, q_mvar=2.0)]
    
    grid = GridTopology(buses=buses, lines=lines, loads=loads)

    # --- B. COMPILE & SOLVE ---
    t0 = time.perf_counter()
    engine = FastGridEngine(grid)
    compile_time = (time.perf_counter() - t0) * 1000
    print(f"[*] Engine Compiled: {compile_time:.2f} ms")

    t1 = time.perf_counter()
    res = engine.solve(loads)
    solve_time = (time.perf_counter() - t1) * 1000
    print(f"[*] Power Flow Solved: {solve_time:.3f} ms")

    # --- C. ANALYZE RESULTS ---
    v_source = abs(res['source']) / 20.0
    v_end = abs(res[end_bus_id]) / 20.0
    print(f"[*] Voltage Drop: {v_source:.4f} p.u. -> {v_end:.4f} p.u.")
    
    if v_end < 0.9:
        print("⚠️ CRITICAL ALERT: End of line undervoltage detected!")

    # --- D. VISUALIZATION ---
    

    dist_km = [i * 1.0 for i in range(len(buses))] # 1km segments
    v_profile = [abs(res[b.id])/20.0 for b in buses]

    plt.figure(figsize=(10, 5))
    plt.plot(dist_km, v_profile, 'o-', markersize=4, label=f'{N_BUSES} Bus Line')
    plt.axhline(0.9, color='red', linestyle='--', label='Lower Limit (0.9 p.u.)')
    plt.fill_between(dist_km, 0.9, 1.1, color='green', alpha=0.1, label='Safe Zone')
    
    plt.xlabel('Distance from Substation (km)')
    plt.ylabel('Voltage (p.u.)')
    plt.title(f'Voltage Profile Analysis ({N_BUSES} Buses)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
