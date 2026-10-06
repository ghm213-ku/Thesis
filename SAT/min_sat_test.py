import SAT.min_sat as min_sat
import networkx as nx
import z3

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
sys.path.append(str(Path(__file__).resolve().parent.parent / "mir_graphs"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "utilities"))

import mir_graphs
import graph_utilities


def test_ilp_accessible():
    m = 8
    iterations = 5
    # Create a simple graph for testing
    graphs = mir_graphs.get_graphs_from_recipes(m)

    # for graph in graphs:
    #     graph_utilities.draw_graph_with_node_labels(graph, title="Original Graph")

    for graph in graphs:
        # print(f"Original Graph: {graph}")
        E_0 = graph_utilities.get_adjacency_matrix(graph_utilities.gen_bell_tree(m))
        E_n = graph_utilities.get_adjacency_matrix(graph)

        print("Building model...")
        opt, dummy, LC, CZ, F, c, p = min_sat.build_quantum_maxsat(
            iterations, m, m, graph.edges()
        )

        print(E_n)

        # print("\n" + "=" * 30)
        # print("      E MATRIX TRACKER")
        # print("=" * 30)
        # for N in range(iterations + 2):
        #     print(f"\n--- Time Step n={N} ---")
        #     for i in range(n):
        #         row = []
        #         for j in range(n):
        #             # Safely grab the variable from the PuLP dictionary
        #             var = model.variablesDict().get(f"E_{N}_{i}_{j}")
        #             val = int(var.varValue) if var and var.varValue is not None else 0
        #             row.append(val)
        #         print(row)

        assert opt.check() == z3.sat, "The model is unsatisfiable."

        operations = graph_utilities.get_operations_sat(
            iterations, m, LC, CZ, F, c, opt.model()
        )
        graph_from_ops = graph_utilities.get_graph_from_operations_sat(
            operations,
            m,
            G=graph_utilities.get_graph_from_adjacency_matrix(E_0),
        )
        is_isomorphic = nx.is_isomorphic(graph, graph_from_ops)
        print(
            f"Is the graph from operations isomorphic to the original graph? {is_isomorphic}"
        )
        assert is_isomorphic, "Graphs are not isomorphic"
