import gurobipy as gp
from gurobipy import GRB

import networkx as nx

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
sys.path.append(str(Path(__file__).resolve().parent.parent / "mir_graphs"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "utilities"))

import mir_graphs
import graph_utilities

# --- Parameters ---
N = 4  # Number of steps (0 to N-1)
M = 4  # Number of nodes (0 to M-1)

target_graph = graph_utilities.get_graph_from_adjacency_matrix(
    [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
)

target_graph_edges = len(list(target_graph.edges()))


def run_checker(sequence):
    """
    Evaluates the sequence of operations.
    Returns a slack value: 0 if target state is reached, > 0 otherwise.

    'sequence' is a list of tuples ordered by step:
    - ('LC', step, i)
    - ('CZ', step, i, j)
    - ('F', step, i, j)
    """
    slack = 1.0  # TODO: Replace with your actual checker logic

    # Example debug print:
    print(f"Checking sequence: {sequence}")

    initial_graph = graph_utilities.gen_bell_tree(M)

    for op in sequence:
        if op[0] == "LC":
            _, step, i = op
            initial_graph = graph_utilities.local_complement(initial_graph, i)
        elif op[0] == "CZ":
            _, step, i, j = op
            initial_graph = graph_utilities.cz_gate_toggle(initial_graph, i, j)
        elif op[0] == "F":
            _, step, i, j = op
            initial_graph = graph_utilities.f_gate(initial_graph, i, j)

    initial_graph_edges = target_graph_edges # len(list(initial_graph.edges()))

    if initial_graph_edges != target_graph_edges:
        return 100.0  # Arbitrary large slack if edge counts don't match
    else:
        if nx.is_isomorphic(initial_graph, target_graph):
            slack = 0.0
        else:
            return 10.0

    return slack


def benders_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        # 1. Retrieve the current integer solution
        active_vars = []
        inactive_vars = []
        sequence = []

        # Extract LC variables
        for (s, i), var in model._y_LC.items():
            if model.cbGetSolution(var) > 0.5:
                active_vars.append(var)
                sequence.append(("LC", s, i))
            else:
                inactive_vars.append(var)

        # Extract CZ and Fusion variables
        for (s, i, j), var in model._y_CZ.items():
            if model.cbGetSolution(var) > 0.5:
                active_vars.append(var)
                sequence.append(("CZ", s, i, j))
            else:
                inactive_vars.append(var)

        for (s, i, j), var in model._y_F.items():
            if model.cbGetSolution(var) > 0.5:
                active_vars.append(var)
                sequence.append(("F", s, i, j))
            else:
                inactive_vars.append(var)

        # Sort sequence by step 's' to ensure chronological order
        sequence.sort(key=lambda x: x[1])

        # 2. Evaluate the sequence
        slack = run_checker(sequence)

        # 3. Add Lazy Cut if the state is not achieved
        if slack > 0:
            # Naive combinatorial Benders cut (No-Good cut)
            # Forces at least one variable in the current layout to flip
            cut_expr = gp.quicksum(1 - var for var in active_vars) + gp.quicksum(
                var for var in inactive_vars
            )

            model.cbLazy(cut_expr >= 1)


def solve_quantum_routing():
    env = gp.Env()
    model = gp.Model("QuantumOps", env=env)

    # Require lazy constraints to enable MIPSOL callbacks
    model.Params.LazyConstraints = 1

    # Pre-calculate valid pairs (i < j to avoid symmetric duplicates, same parity)
    valid_pairs = [(i, j) for i in range(M) for j in range(i + 1, M) if i % 2 == j % 2]

    # --- Variables ---
    y_LC = {}
    for s in range(N):
        for i in range(M):
            y_LC[s, i] = model.addVar(vtype=GRB.BINARY, name=f"LC_{s}_{i}")

    y_CZ = {}
    y_F = {}
    for s in range(N):
        for i, j in valid_pairs:
            y_CZ[s, i, j] = model.addVar(vtype=GRB.BINARY, name=f"CZ_{s}_{i}_{j}")
            y_F[s, i, j] = model.addVar(vtype=GRB.BINARY, name=f"F_{s}_{i}_{j}")

    # Store variable dictionaries in the model object for callback access
    model._y_LC = y_LC
    model._y_CZ = y_CZ
    model._y_F = y_F

    # --- Objective ---
    obj = (
        gp.quicksum(3.17 * var for var in y_CZ.values())
        + gp.quicksum(1.0 * var for var in y_F.values())
        + gp.quicksum(0.01 * var for var in y_LC.values())
    )

    model.setObjective(obj, GRB.MINIMIZE)

    # --- Structural Constraints ---
    # A node can participate in AT MOST ONE operation per step
    # for s in range(N):
    #     for node in range(M):
    #         expr = y_LC[s, node]
    #         # Add CZ operations involving this node
    #         expr += gp.quicksum(
    #             y_CZ[s, i, j] for (i, j) in valid_pairs if i == node or j == node
    #         )
    #         # Add Fusion operations involving this node
    #         expr += gp.quicksum(
    #             y_F[s, i, j] for (i, j) in valid_pairs if i == node or j == node
    #         )

    #         model.addConstr(expr <= 1, name=f"OneOpPerNode_{s}_{node}")

    for n in range(N):
        step_ops = (
            gp.quicksum(y_LC[key] for key in y_LC.keys() if key[0] == n)
            + gp.quicksum(y_CZ[key] for key in y_CZ.keys() if key[0] == n)
            + gp.quicksum(y_F[key] for key in y_F.keys() if key[0] == n)
        )
        # Use == 1 if you want exactly one operation, or <= 1 for at most one
        model.addConstr(step_ops <= 1, name=f"ExactlyOneOp_step_{n}")

    # --- Optimize ---
    model.optimize(benders_callback)

    # --- Results ---
    if model.Status == GRB.OPTIMAL:
        print("\nOptimal sequence found:")
        for (s, i), var in y_LC.items():
            if var.X > 0.5:
                print(f"Step {s}: LC on {i}")
        for (s, i, j), var in y_CZ.items():
            if var.X > 0.5:
                print(f"Step {s}: CZ on {i},{j}")
        for (s, i, j), var in y_F.items():
            if var.X > 0.5:
                print(f"Step {s}: Fusion on {i},{j}")
    elif model.Status == GRB.INFEASIBLE:
        print(
            "\nProblem is infeasible. No sequence achieves slack == 0 within N steps."
        )


if __name__ == "__main__":
    solve_quantum_routing()
