import ilp
import ilp_copy
import ilp_latest
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

ilps = [ilp, ilp_copy, ilp_latest]

current_ilp = ilp_latest


def test_ilp_accessible():
    n = 4
    iterations = 5
    # Create a simple graph for testing
    graphs = mir_graphs.get_graphs_from_recipes(n)

    for graph in graphs:
        # print(f"Original Graph: {graph}")
        E_0 = graph_utilities.get_adjacency_matrix(graph_utilities.gen_bell_tree(n))
        L = graph_utilities.get_leader_matrix(n)
        E_n = graph_utilities.get_adjacency_matrix(graph)

        model = current_ilp.build_model(iterations, n - 1, E_0, E_n, L)

        print(
            f"ILP model has {model.numVariables()} variables and {model.numConstraints()} constraints."
        )

        model.solve(pulp.GUROBI(msg=0))

        if model.status == pulp.LpStatusOptimal:
            operations = graph_utilities.get_operations(model)
            sorted_operations = graph_utilities.get_sorted_operations(operations)
            node_deletion_ops = graph_utilities.get_node_deletion_operations(model)
            graph_from_ops = graph_utilities.get_graph_from_operations(
                sorted_operations,
                node_deletion_ops,
                n,
                G=graph_utilities.get_graph_from_adjacency_matrix(E_0),
            )
            is_isomorphic = nx.is_isomorphic(graph, graph_from_ops)
            print(
                f"Is the graph from operations isomorphic to the original graph? {is_isomorphic}"
            )
            assert is_isomorphic, "Graphs are not isomorphic"
        else:
            print(
                f"Failed to find optimal solution. Status: {pulp.LpStatus[model.status]}"
            )


def test_ilp_LC():
    n = 6
    iterations = 5
    # Create a simple graph for testing
    E_0 = [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    T = [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]
    L_0 = [
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    # Apply the ILP to find the sequence of operations
    model = current_ilp.build_model(iterations, n - 1, E_0, T, L_0)

    model.solve(pulp.GUROBI(msg=0))

    if model.status == pulp.LpStatusOptimal:
        operations = graph_utilities.get_operations(model)
        sorted_operations = graph_utilities.get_sorted_operations(operations)
        node_deletion_ops = graph_utilities.get_node_deletion_operations(model)
        graph_from_ops = graph_utilities.get_graph_from_operations(
            sorted_operations,
            node_deletion_ops,
            n,
            G=graph_utilities.get_graph_from_adjacency_matrix(E_0),
        )
        is_isomorphic = nx.is_isomorphic(
            graph_utilities.get_graph_from_adjacency_matrix(T), graph_from_ops
        )
        print(
            f"Is the graph from operations isomorphic to the original graph? {is_isomorphic}"
        )
        assert is_isomorphic, "Graphs are not isomorphic"
    else:
        print(f"Failed to find optimal solution. Status: {pulp.LpStatus[model.status]}")


def test_ilp_node_deletion():
    n = 6
    iterations = 5
    # Create a simple graph for testing
    E_0 = [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    T = [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    L_0 = [
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    # Apply the ILP to find the sequence of operations
    model = current_ilp.build_model(iterations, n - 1, E_0, T, L_0)

    model.solve(pulp.GUROBI(msg=0))

    if model.status == pulp.LpStatusOptimal:
        operations = graph_utilities.get_operations(model)
        sorted_operations = graph_utilities.get_sorted_operations(operations)
        node_deletion_ops = graph_utilities.get_node_deletion_operations(model)
        graph_from_ops = graph_utilities.get_graph_from_operations(
            sorted_operations,
            node_deletion_ops,
            n,
            G=graph_utilities.get_graph_from_adjacency_matrix(E_0),
        )
        # )
        is_isomorphic = nx.is_isomorphic(
            graph_utilities.get_graph_from_adjacency_matrix(T), graph_from_ops
        )
        print(
            f"Is the graph from operations isomorphic to the original graph? {is_isomorphic}"
        )
        assert is_isomorphic, "Graphs are not isomorphic"
    else:
        print(f"Failed to find optimal solution. Status: {pulp.LpStatus[model.status]}")


def test_ilp_LC_node_deletion():
    n = 6
    iterations = 5
    # Create a simple graph for testing
    E_0 = [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    T = [
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]
    L_0 = [
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    # Apply the ILP to find the sequence of operations
    model = current_ilp.build_model(iterations, n - 1, E_0, T, L_0)

    model.solve(pulp.GUROBI(msg=0))

    if model.status == pulp.LpStatusOptimal:
        operations = graph_utilities.get_operations(model)
        sorted_operations = graph_utilities.get_sorted_operations(operations)
        node_deletion_ops = graph_utilities.get_node_deletion_operations(model)
        graph_from_ops = graph_utilities.get_graph_from_operations(
            sorted_operations,
            node_deletion_ops,
            n,
            G=graph_utilities.get_graph_from_adjacency_matrix(E_0),
        )
        is_isomorphic = nx.is_isomorphic(
            graph_utilities.get_graph_from_adjacency_matrix(T), graph_from_ops
        )
        print(
            f"Is the graph from operations isomorphic to the original graph? {is_isomorphic}"
        )
        assert is_isomorphic, "Graphs are not isomorphic"
    else:
        print(f"Failed to find optimal solution. Status: {pulp.LpStatus[model.status]}")
