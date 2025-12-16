import numpy as np
import networkx as nx
import unittest
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Literal, Optional, Dict
from scipy.sparse import csr_matrix

# ==========================================
# 1. DOMAIN DATA LAYER (Pydantic / CIM)
# ==========================================
class Bus(BaseModel):
    """
    Represents a Topological Node (ConnectivityNode).
    Physical Constraint: Voltage must be positive.
    """
    id: str = Field(..., description="Unique mRID")
    voltage_level_kv: float = Field(..., gt=0, description="Nominal voltage")
    type: Literal['PQ', 'PV', 'SLACK'] = 'PQ'

    @field_validator('voltage_level_kv')
    @classmethod
    def check_standard_voltage(cls, v: float) -> float:
        # Soft validation for standard European distribution voltages
        standards = [0.4, 10, 20, 110, 220, 380]
        if v not in standards:
            # In a real logger, this would be a warning, not a print
            pass 
        return v

class ACLineSegment(BaseModel):
    """
    Represents a distribution line/cable with impedance attributes.
    Ref: IEC 61970 ACLineSegment.
    """
    id: str
    from_bus: str
    to_bus: str
    length_km: float = Field(..., gt=0, description="Zero length causes singular Jacobian")
    r_ohm_per_km: float = Field(..., ge=0)
    x_ohm_per_km: float = Field(..., ge=0)
    c_nf_per_km: float = Field(default=0.0, ge=0)
    max_current_a: float = Field(..., gt=0)

    @property
    def total_impedance(self) -> complex:
        """Z = (R + jX) * Length"""
        return complex(self.r_ohm_per_km, self.x_ohm_per_km) * self.length_km

    @property
    def total_admittance(self) -> complex:
        """Y_shunt = j * 2*pi*f * C * Length"""
        omega = 2 * np.pi * 50.0
        b_tot = omega * (self.c_nf_per_km * 1e-9) * self.length_km
        return complex(0, b_tot)

class Load(BaseModel):
    """Represents a consumer or prosumer connected to a bus."""
    id: str
    bus: str
    p_mw: float
    q_mvar: float

class GridModel(BaseModel):
    """
    Container for the entire grid topology. 
    Enforces referential integrity between components.
    """
    buses: List[Bus]
    lines: List[ACLineSegment]
    loads: List[Load] = []

    @model_validator(mode='after')
    def check_topology_integrity(self):
        """Ensures all lines connect to existing buses (Fail Fast principle)."""
        bus_ids = {b.id for b in self.buses}
        for line in self.lines:
            if line.from_bus not in bus_ids:
                raise ValueError(f"Line {line.id} connects to unknown Bus {line.from_bus}")
            if line.to_bus not in bus_ids:
                raise ValueError(f"Line {line.id} connects to unknown Bus {line.to_bus}")
        return self

    def get_bus(self, bus_id: str) -> Bus:
        return next(b for b in self.buses if b.id == bus_id)

# ==========================================
# 2. TOPOLOGY & GRAPH THEORY LAYER
# ==========================================
class TopologyService:
    @staticmethod
    def build_graph(grid: GridModel) -> nx.Graph:
        """Converts GridModel to NetworkX graph for traversal."""
        G = nx.Graph()
        for bus in grid.buses:
            G.add_node(bus.id, obj=bus)
        
        for line in grid.lines:
            weight = abs(line.total_impedance)
            G.add_edge(line.from_bus, line.to_bus, id=line.id, obj=line, weight=weight)
        return G

    @staticmethod
    def validate_radiality(G: nx.Graph):
        """
        Validates that the grid is a Forest (collection of disjoint Trees).
        Distribution grids must be radial for protection coordination.
        """
        if not nx.is_forest(G):
            try:
                cycles = nx.cycle_basis(G)
                raise ValueError(f"Grid contains {len(cycles)} loops. Must be radial.")
            except nx.NetworkXNoCycle:
                 pass

    @staticmethod
    def get_bfs_ordering(G: nx.Graph, root_id: str) -> List[str]:
        """
        Returns topological ordering using BFS. 
        Essential for determining flow direction in FBS.
        """
        if root_id not in G.nodes:
            raise ValueError("Root node not found in grid.")
        return list(nx.topological_sort(nx.bfs_tree(G, source=root_id)))

# ==========================================
# 3. CALCULATION CORE (SOLVERS & CONTROL)
# ==========================================
class PowerFlowSolver:
    @staticmethod
    def solve_fbs(grid: GridModel, root_id: str, tol=1e-4, max_iter=20) -> Dict[str, complex]:
        """
        Forward-Backward Sweep (FBS) Algorithm.
        Stable for high R/X ratio distribution grids.
        """
        G = TopologyService.build_graph(grid)
        TopologyService.validate_radiality(G)
        order = TopologyService.get_bfs_ordering(G, root_id)
        
        # Initialization (Flat Start)
        voltages = {b.id: complex(b.voltage_level_kv, 0) for b in grid.buses}
        
        for _ in range(max_iter):
            max_v_diff = 0.0
            
            # --- 1. Load Injection Calculation ---
            # I_load = (S / V)*
            node_currents = {node: 0j for node in order}
            for load in grid.loads:
                v_node = voltages[load.bus]
                if abs(v_node) > 1e-6:
                    s_load = complex(load.p_mw, load.q_mvar)
                    node_currents[load.bus] += (s_load / v_node).conjugate()

            # --- 2. Backward Sweep & Forward Sweep Combined ---
            
            # Forward Sweep (Voltage Update)
            for node_id in order:
                if node_id == root_id: continue
                
                # Identify Parent (upstream neighbor)
                neighbors = list(G.neighbors(node_id))
                parent_id = next(n for n in neighbors if order.index(n) < order.index(node_id))
                
                # Get Line Impedance
                line = G.edges[parent_id, node_id]['obj']
                z_line = line.total_impedance
                
                # Approximate Branch Current (I_branch ~ I_load for leaf nodes)
                i_branch = node_currents[node_id] 
                
                # V_child = V_parent - I * Z
                v_new = voltages[parent_id] - (i_branch * z_line)
                
                max_v_diff = max(max_v_diff, abs(v_new - voltages[node_id]))
                voltages[node_id] = v_new
            
            if max_v_diff < tol:
                return voltages

        print("Warning: FBS did not converge.")
        return voltages

class FuzzyVoltVarController:
    """
    Implements a Fuzzy Logic Controller (FLC) for Volt-VAR Optimization.
    
    USE CASE: 
    To determine the optimal Reactive Power (Q) injection for a PV inverter 
    based on the local grid voltage.
    
    WHY IT IS IMPORTANT:
    - Non-Linearity: Grid voltage response is non-linear; Fuzzy logic handles this 
      better than simple linear droop control.
    - Uncertainty: Works well even if grid impedance data is imperfect.
    """

    def __init__(self):
        # Linguistic variables map "Crisp" numbers to "Concepts"
        pass

    def _membership_triangle(self, x: float, left: float, center: float, right: float) -> float:
        """
        STEP 1: MEMBERSHIP FUNCTION (Fuzzification Helper)
        Calculates 'how much' a value belongs to a category (0.0 to 1.0).
        """
        if x <= left or x >= right:
            return 0.0
        elif x == center:
            return 1.0
        elif x < center:
            return (x - left) / (center - left)
        else:
            return (right - x) / (right - center)

    def fuzzify_voltage_error(self, error: float) -> Dict[str, float]:
        """
        STEP 2: FUZZIFICATION (Input Processing)
        Converts voltage error (V_ref - V_meas) into linguistic terms:
        - NB (Negative Big): Voltage high -> Need to lower.
        - Z (Zero): Voltage nominal.
        - PB (Positive Big): Voltage low -> Need to boost.
        """
        mu = {}
        # Voltage is WAY too high (Error is Negative)
        mu['NB'] = self._membership_triangle(error, -0.15, -0.10, -0.0) 
        # Voltage is nominal (Error is Zero)
        mu['Z']  = self._membership_triangle(error, -0.05,  0.00,  0.05)
        # Voltage is WAY too low (Error is Positive)
        mu['PB'] = self._membership_triangle(error,  0.00,  0.10,  0.15)
        return mu

    def evaluate_rules(self, mu: Dict[str, float]) -> Dict[str, float]:
        """
        STEP 3: INFERENCE ENGINE (Rule Evaluation)
        Rule 1: If Voltage High (Error NB) -> Absorb Q (Output NB)
        Rule 2: If Voltage Normal (Error Z) -> Do Nothing (Output Z)
        Rule 3: If Voltage Low (Error PB) -> Inject Q (Output PB)
        """
        output_strength = {}
        output_strength['NB'] = mu['NB'] 
        output_strength['Z']  = mu['Z']  
        output_strength['PB'] = mu['PB'] 
        return output_strength

    def defuzzify(self, output_strength: Dict[str, float]) -> float:
        """
        STEP 4: DEFUZZIFICATION (Output Generation)
        Converts fuzzy conclusions to inverter command (Q in MVar) using Centroid method.
        """
        # Centers of output functions (MVar): Absorb, Idle, Inject
        centers = {'NB': -1.0, 'Z': 0.0, 'PB': 1.0}
        
        numerator = 0.0
        denominator = 0.0
        
        for term, strength in output_strength.items():
            numerator += strength * centers[term]
            denominator += strength
            
        if denominator == 0:
            return 0.0
            
        return numerator / denominator

    def compute_q_setpoint(self, v_measured_pu: float) -> float:
        """
        MAIN EXECUTION PIPELINE for the Controller.
        """
        # 1. Calculate Error (Reference is usually 1.0 p.u.)
        error = 1.0 - v_measured_pu
        
        # 2. Fuzzify
        fuzzy_inputs = self.fuzzify_voltage_error(error)
        
        # 3. Apply Rules
        fuzzy_outputs = self.evaluate_rules(fuzzy_inputs)
        
        # 4. Defuzzify to get Crisp Output
        q_mvar = self.defuzzify(fuzzy_outputs)
        
        return q_mvar

# ==========================================
# 4. APPLICATION LAYER (HOSTING CAPACITY)
# ==========================================
class AnalyticsEngine:
    @staticmethod
    def hosting_capacity(grid: GridModel, target_node: str, solver_func) -> float:
        """
        Determines max generation (MW) connectable to target_node using Binary Search.
        """
        min_p, max_p = 0.0, 50.0 # MW
        precision = 0.01 
        
        while (max_p - min_p) > precision:
            mid_p = (min_p + max_p) / 2
            
            # Inject Test Generation (Negative Load)
            grid.loads.append(Load(id="gen_test", bus=target_node, p_mw=-mid_p, q_mvar=0))
            
            valid = False
            try:
                results = solver_func(grid, root_id="source_bus")
                # Check Voltage Constraint (0.9 - 1.1 pu)
                v_pu = abs(results[target_node]) / grid.get_bus(target_node).voltage_level_kv
                if 0.9 <= v_pu <= 1.1:
                    valid = True
            except Exception:
                pass # Divergence is a failure
            
            if valid:
                min_p = mid_p 
            else:
                max_p = mid_p 
                
            grid.loads.pop() # Cleanup
            
        return min_p

# ==========================================
# 5. VISUALIZATION LAYER (MATPLOTLIB)
# ==========================================
class GridVisualizer:
    @staticmethod
    def plot_voltage_profile(grid: GridModel, base_res: dict, hc_res: dict, hc_mw: float):
        """
        Plots the voltage profile (Distance vs Voltage p.u.).
        """
        # Data Extraction: Calculate approximate distance from source
        dist_map = {"source_bus": 0.0}
        cum_dist = 0.0
        
        # Traverse lines to map distances (Demo specific)
        for line in grid.lines:
            cum_dist += line.length_km
            dist_map[line.to_bus] = cum_dist
            
        nodes = list(dist_map.keys())
        dists = [dist_map[n] for n in nodes]
        
        # Extract Voltages and normalize to p.u.
        v_base_pu = [abs(base_res[n]) / grid.get_bus(n).voltage_level_kv for n in nodes]
        v_hc_pu = [abs(hc_res[n]) / grid.get_bus(n).voltage_level_kv for n in nodes]
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        # 1. Plot Limits
        plt.axhline(1.1, color='r', linestyle='--', label='Upper Limit (1.1 p.u.)')
        plt.axhline(0.9, color='r', linestyle='--', label='Lower Limit (0.9 p.u.)')
        plt.axhline(1.0, color='k', linewidth=0.5, alpha=0.5)
        
        # 2. Plot Profiles
        plt.plot(dists, v_base_pu, 'o-', label='Base Case (Load Only)', color='blue', linewidth=2)
        plt.plot(dists, v_hc_pu, 's-', label=f'With {hc_mw:.2f} MW Gen', color='green', linewidth=2)
        
        plt.title(f"Feeder Voltage Profile: Hosting Capacity Analysis")
        plt.xlabel("Distance from Substation (km)")
        plt.ylabel("Voltage (p.u.)")
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

# ==========================================
# 6. VALIDATION SUITE (UNIT TESTS)
# ==========================================
class GridPhysicsTest(unittest.TestCase):
    """
    Validates the physical fidelity of the solver.
    Uses 'Trivial Network' strategy.
    """
    
    def setUp(self):
        # 2-Node System: Source (10kV) --[1 Ohm]--> Load (10MW)
        self.buses = [
            Bus(id="source_bus", voltage_level_kv=10.0, type='SLACK'),
            Bus(id="load_bus", voltage_level_kv=10.0, type='PQ')
        ]
        self.lines = [
            ACLineSegment(
                id="line_1", from_bus="source_bus", to_bus="load_bus",
                length_km=1.0, r_ohm_per_km=1.0, x_ohm_per_km=0.0, max_current_a=2000
            )
        ]
        self.grid = GridModel(buses=self.buses, lines=self.lines)

    def test_voltage_drop_approx(self):
        """
        Tests if V_load drops correctly under load.
        """
        self.grid.loads = [Load(id="load_1", bus="load_bus", p_mw=10.0, q_mvar=0.0)]
        
        voltages = PowerFlowSolver.solve_fbs(self.grid, root_id="source_bus")
        v_load_mag = abs(voltages["load_bus"])
        
        print(f"\n[Test] Manual Calc Check: Expected ~9.0kV, Got {v_load_mag:.3f}kV")
        self.assertTrue(8.5 < v_load_mag < 9.5, "Voltage drop physics failed")

    def test_kirchhoff_law(self):
        """
       Verifies Energy Conservation: P_source = P_load + P_loss.
        """
        self.grid.loads = [Load(id="load_1", bus="load_bus", p_mw=5.0, q_mvar=0.0)]
        res = PowerFlowSolver.solve_fbs(self.grid, root_id="source_bus")
        
        # Calculate Flows
        v_src = res["source_bus"]
        v_load = res["load_bus"]
        current = (v_src - v_load) / self.grid.lines[0].total_impedance
        
        p_loss = (abs(current)**2) * self.grid.lines[0].total_impedance.real
        p_load = 5.0 # MW
        
        # Source injection calculation (complex power)
        s_source = v_src * current.conjugate()
        p_source = s_source.real
        
        print(f"[Test] KCL Check: Source {p_source:.3f}MW vs (Load {p_load} + Loss {p_loss:.3f})MW")
        self.assertAlmostEqual(p_source, p_load + p_loss, places=2)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Run Verification Suite
    print("--- 1. Executing Physics Validation Suite ---")
    suite = unittest.TestLoader().loadTestsFromTestCase(GridPhysicsTest)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    
    if result.wasSuccessful():
        print("\n--- 2. Running Application Demo (Hosting Capacity) ---")
        
        # Define Production Grid (5km Line)
        buses = [
            Bus(id="source_bus", voltage_level_kv=10.0, type='SLACK'),
            Bus(id="load_bus", voltage_level_kv=10.0, type='PQ')
        ]
        lines = [
            ACLineSegment(
                id="line_1", from_bus="source_bus", to_bus="load_bus",
                length_km=5.0, r_ohm_per_km=0.1, x_ohm_per_km=0.1, max_current_a=400
            )
        ]
        grid = GridModel(buses=buses, lines=lines)
        
        # A. Run Base Case (Just Load)
        grid.loads = [Load(id="base_load", bus="load_bus", p_mw=2.0, q_mvar=0)]
        res_base = PowerFlowSolver.solve_fbs(grid, "source_bus")
        
        # B. Run Hosting Capacity Analysis
        hc_mw = AnalyticsEngine.hosting_capacity(grid, "load_bus", PowerFlowSolver.solve_fbs)
        print(f"Calculated Hosting Capacity: {hc_mw:.2f} MW")
        
        # C. Run HC Case (Load + Calculated Max Generation)
        grid.loads.append(Load(id="max_gen", bus="load_bus", p_mw=-hc_mw, q_mvar=0))
        res_hc = PowerFlowSolver.solve_fbs(grid, "source_bus")
        
        print("\n--- 3. Running Fuzzy Volt-VAR Control Demo ---")
        # Instantiate Controller
        fvc = FuzzyVoltVarController()
        
        # Simulate high voltage scenario (e.g., from the HC case above)
        v_hc_pu = abs(res_hc['load_bus']) / 10.0 # Normalize to p.u.
        q_output = fvc.compute_q_setpoint(v_hc_pu)
        
        print(f"Scenario: Grid Voltage is {v_hc_pu:.3f} p.u. (High Generation)")
        print(f"Fuzzy Controller Action: Inverter should {'Inject' if q_output > 0 else 'Absorb'} {abs(q_output):.3f} MVar")
        
        # D. Visualize Results
        print("\nGenerating Voltage Profile Plot...")
        GridVisualizer.plot_voltage_profile(grid, res_base, res_hc, hc_mw)
    else:
        print("\n[CRITICAL] Validation Failed. Aborting Application Start.")
