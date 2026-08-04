import networkx as nx
import queue

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
sys.path.append(str(Path(__file__).resolve().parent.parent / "utilities"))
import graph_utilities as graph_utilities
import visualizer as visualizer


def get_signature(graph: nx.Graph) -> tuple:
    """
    Get a unique signature for the graph based on its nodes and edges.

    Args:
        graph: The graph for which to get the signature.

    Returns:
        A tuple containing sorted nodes and sorted edges of the graph.
    """
    return (tuple(sorted(graph.nodes())), tuple(sorted(graph.edges())))


def get_node_index(graph: nx.Graph, graph_index: dict) -> int:
    """
    Get the index of a graph in the graph index.

    Args:
        graph: The graph for which to get the index.
        graph_index: A dictionary mapping graphs to their indices in the meta graph.
    """
    signature = get_signature(graph)
    return graph_index.get(signature, -1)  # Return -1 if the graph is not found


def add_graph_to_meta(
    meta_graph: nx.Graph, graph_node: nx.Graph, graph_index: dict
) -> bool:
    """
    Add a graph to the meta graph and update the graph index.

    Args:
        meta_graph: The meta graph to which the new graph will be added.
        graph_node: The new graph to be added.
        graph_index: A dictionary mapping graphs to their indices in the meta graph.
    """
    signature = get_signature(graph_node)

    if signature in graph_index:
        return False  # Graph already exists in the meta graph

    meta_graph.add_node(len(graph_index), graph=graph_node)
    graph_index[signature] = len(graph_index)
    return True


def is_edge_present(
    meta_graph: nx.Graph,
    graph_index: dict,
    graph1: nx.Graph,
    graph2: nx.Graph,
    weight: float = 0.0,
    operation: str = "",
):
    """
    Check if an edge is present between two graphs in the meta graph. Adds the edge if it is not present.

    Args:
        meta_graph: The meta graph to which the edge will be added.
        graph_index: A dictionary mapping graphs to their indices in the meta graph.
        graph1: The first graph.
        graph2: The second graph.
        weight: The weight of the edge (default is 0).
        operation: The operation associated with the edge (default is an empty string).
    """

    add_graph_to_meta(meta_graph, graph2, graph_index)

    index1 = get_node_index(graph1, graph_index)
    index2 = get_node_index(graph2, graph_index)

    assert (
        index1 != -1 and index2 != -1
    ), "One or both graphs are not present in the meta graph."

    if (
        meta_graph.has_edge(index1, get_node_index(graph2, graph_index))
        or index1 == index2
    ):
        return True  # Edge already exists or self-loop

    meta_graph.add_edge(index1, index2, weight=weight, operation=operation)
    if operation.startswith("  ") or operation.startswith("CZ"):
        meta_graph.add_edge(index2, index1, weight=weight, operation=operation)

    return False  # Edge was not present and has been added


def queue_two_qubit_operations(
    meta_graph: nx.Graph,
    graph_index: dict,
    q: queue.Queue,
    current_graph: nx.Graph,
    operations: list,
):
    """
    Queue the two-qubit operations for processing.

    Args:
        meta_graph: The meta graph to which the new graphs will be added.
        graph_index: A dictionary mapping graphs to their indices in the meta graph.
        q: The queue to which the new graphs will be added.
        current_graph: The current graph being processed.
        operations: A list of two-qubit operations to apply.
    """
    if len(operations) == 0:
        return  # No operations to apply

    remaining_ops = []  # Remaining operations after the first one

    if len(operations) > 1:
        remaining_ops = [operations[1]]  # All operations except the first one

    # Apply CZ gate
    new_graph_cz = graph_utilities.cz_gate_toggle(
        current_graph, operations[0][0], operations[0][1]
    )
    if not is_edge_present(
        meta_graph,
        graph_index,
        current_graph,
        new_graph_cz,
        3.17,
        f"CZ({operations[0][0]},{operations[0][1]})",
    ):
        q.put((new_graph_cz, remaining_ops))

    # Apply F gate
    new_graph_f = graph_utilities.f_gate(
        current_graph, operations[0][0], operations[0][1]
    )
    if not is_edge_present(
        meta_graph,
        graph_index,
        current_graph,
        new_graph_f,
        1,
        f"F({operations[0][0]},{operations[0][1]})",
    ):
        q.put((new_graph_f, remaining_ops))

        # Apply F gate in the other direction
        new_graph_f = graph_utilities.f_gate(
            current_graph, operations[0][1], operations[0][0]
        )
        if not is_edge_present(
            meta_graph,
            graph_index,
            current_graph,
            new_graph_f,
            1,
            f"F({operations[0][1]},{operations[0][0]})",
        ):
            q.put((new_graph_f, remaining_ops))

    if len(operations) > 1:
        remaining_ops = [operations[1]]  # All operations except the first two
        # Apply CZ gate
        new_graph_cz = graph_utilities.cz_gate_toggle(
            current_graph, operations[1][0], operations[1][1]
        )
        if not is_edge_present(
            meta_graph,
            graph_index,
            current_graph,
            new_graph_cz,
            3.17,
            f"CZ({operations[1][0]},{operations[1][1]})",
        ):
            q.put((new_graph_cz, remaining_ops))

        # Apply F gate
        new_graph_f = graph_utilities.f_gate(
            current_graph, operations[1][0], operations[1][1]
        )
        if not is_edge_present(
            meta_graph,
            graph_index,
            current_graph,
            new_graph_f,
            1,
            f"F({operations[1][0]},{operations[1][1]})",
        ):
            q.put((new_graph_f, remaining_ops))

        # Apply F gate in the other direction
        new_graph_f = graph_utilities.f_gate(
            current_graph, operations[1][1], operations[1][0]
        )
        if not is_edge_present(
            meta_graph,
            graph_index,
            current_graph,
            new_graph_f,
            1,
            f"F({operations[1][1]},{operations[1][0]})",
        ):
            q.put((new_graph_f, remaining_ops))


def create_graph(n: int) -> tuple[nx.Graph, dict]:
    """
    Create the graph consisting of all the paths and nodes for shortest distance trial.
    """
    q = queue.Queue()
    graph_index = {}
    G = graph_utilities.gen_bell_tree(n)
    meta_graph = nx.DiGraph()
    operations = [[0, 2], [1, 3]]  # List of operations to apply

    q.put((G, operations))  # Graph , gates,
    add_graph_to_meta(meta_graph, G, graph_index)

    while not q.empty():
        current_graph, operations = q.get()

        queue_two_qubit_operations(
            meta_graph, graph_index, q, current_graph, operations
        )

        # All local complementations
        for i in range(n):
            new_graph = graph_utilities.local_complement(current_graph, i)
            if not is_edge_present(
                meta_graph, graph_index, current_graph, new_graph, 0, f"LC({i})"
            ):
                q.put((new_graph, operations))

    return tuple([meta_graph, graph_index])


if __name__ == "__main__":
    n = 4  # Number of qubits
    source_node = 0
    target_graph = nx.Graph()
    target_graph.add_nodes_from(range(n))
    target_graph.add_edges_from([(0, 2), (2, 1), (1, 3), (3, 0)])

    meta_graph, graph_index = create_graph(n)
    target_node_index = get_node_index(target_graph, graph_index)
    path = nx.shortest_path(
        meta_graph, source=source_node, target=target_node_index, weight="weight"
    )

    for i in range(len(path) - 1):
        print(
            (
                f"Edge from {path[i]} to {path[i+1]} with operation: {meta_graph.edges[path[i], path[i+1]]['operation']} and weight: {meta_graph.edges[path[i], path[i+1]]['weight']}"
            )
        )

    visualizer.run_dashboard(meta_graph, port=8050)
