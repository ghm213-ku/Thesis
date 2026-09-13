from cffi import model

import ilp
import pulp
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


def get_leader_matrix(n: int):
    L = [[0] * n for _ in range(n)]
    for i in range(0, n, 2):
        L[i][i] = 1
        L[i][i + 1] = 1
    return L


def get_adjacency_matrix(graph):
    adj = nx.adjacency_matrix(graph)
    matrix = [[0] * graph.number_of_nodes() for _ in range(graph.number_of_nodes())]
    for i in range(graph.number_of_nodes()):
        for j in range(graph.number_of_nodes()):
            matrix[i][j] = int(adj[i, j])
            if i >= j:
                matrix[i][j] = 0
    return matrix


def get_operations(model):
    operations = []
    for v in model.variables():
        if (
            v.name.startswith("y_")
            and v.varValue is not None
            and v.varValue > 0.5
            and not ("active" in v.name)  # Exclude active sum variables
            and not ("dummy" in v.name)  # Exclude dummy variables
        ):
            operations.append(v.name)
    return operations


def get_sorted_operations(operations):
    # Sort operations based on the first number in the operation name
    return sorted(operations, key=lambda op: int(op.split("_")[2]))


def get_graph_from_operations(operations, n):
    # Create an empty graph with n nodes
    G = graph_utilities.gen_bell_tree(n)

    # Apply operations to the graph
    for op in operations:
        parts = op.split("_")
        op_type = parts[1]
        node1 = int(parts[3])
        node2 = int(parts[4]) if len(parts) > 4 else None

        if op_type == "CZ":
            G = graph_utilities.cz_gate_toggle(G, node1, node2)
        elif op_type == "F":
            G = graph_utilities.f_gate(G, node1, node2)
        else:
            G = graph_utilities.local_complement(G, node1)
    return G


def test_ilp():
    n = 8
    iterations = 5
    # Create a simple graph for testing
    graphs = mir_graphs.get_graphs_from_recipes(n)

    for graph in graphs:
        # print(f"Original Graph: {graph}")
        E_0 = get_adjacency_matrix(graph_utilities.gen_bell_tree(n))
        L = get_leader_matrix(n)
        E_n = get_adjacency_matrix(graph)

        print(f"Adjacency Matrix of Original Graph (E_n)")
        for i in range(n):
            print(E_n[i])

        # print(f"Adjacency Matrix of Initial Graph (E_0)")
        # for i in range(n):
        #     print(E_0[i])

        # print(f"Leader Matrix (L)")
        # for i in range(n):
        #     print(L[i])

        model = ilp.build_model(iterations, n - 1, E_0, E_n, L)

        model.solve(pulp.GUROBI(msg=0))

        if model.status == pulp.LpStatusOptimal:
            operations = get_operations(model)
            sorted_operations = get_sorted_operations(operations)
            graph_from_ops = get_graph_from_operations(sorted_operations, n)
            is_isomorphic = nx.is_isomorphic(graph, graph_from_ops)
            print(
                f"Is the graph from operations isomorphic to the original graph? {is_isomorphic}"
            )
            assert is_isomorphic, "Graphs are not isomorphic"
        else:
            print(
                f"Failed to find optimal solution. Status: {pulp.LpStatus[model.status]}"
            )


test_ilp()
