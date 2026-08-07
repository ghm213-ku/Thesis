import networkx as nx
import queue
from itertools import count

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
sys.path.append(str(Path(__file__).resolve().parent.parent / "utilities"))
import graph_utilities as graph_utilities
import visualizer as visualizer
from shortest_distance_utilities import (
    get_signature,
    get_node_index,
    print_shortest_path,
    get_highlighted_items_from_path,
    is_path_valid,
    is_op_sequence_valid,
    shortest_path,
)


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

    if index1 == index2 or meta_graph.has_edge(index1, index2):
        return True  # Self loop

    meta_graph.add_edge(index1, index2, weight=weight, operation=operation)
    # if operation.startswith("  ") or operation.startswith("CZ"):
    #     meta_graph.add_edge(index2, index1, weight=weight, operation=operation)

    return False  # Edge was not present and has been added


def queue_operation(
    meta_graph: nx.Graph,
    graph_index: dict,
    q: queue.Queue,
    current_graph: nx.Graph,
    operation: str,
    remaining_ops: list,
):
    """
    Queue a single two-qubit operation for processing.

    Args:
        meta_graph: The meta graph to which the new graphs will be added.
        graph_index: A dictionary mapping graphs to their indices in the meta graph.
        q: The queue to which the new graphs will be added.
        current_graph: The current graph being processed.
        operation: Operation string.
    """
    if operation.startswith("CZ"):
        i, j = map(int, operation[3:-1].split(","))
        new_graph = graph_utilities.cz_gate_toggle(current_graph, i, j)
        if not is_edge_present(
            meta_graph,
            graph_index,
            current_graph,
            new_graph,
            3.17,
            operation,
        ):
            q.put(new_graph, remaining_ops)
    elif operation.startswith("F"):
        i, j = map(int, operation[2:-1].split(","))
        new_graph = graph_utilities.f_gate(current_graph, i, j)
        if not is_edge_present(
            meta_graph,
            graph_index,
            current_graph,
            new_graph,
            1,
            operation,
        ):
            q.put(new_graph, remaining_ops)
    elif operation.startswith("LC"):
        i = int(operation[3:-1])
        new_graph = graph_utilities.local_complement(current_graph, i)
        if not is_edge_present(
            meta_graph,
            graph_index,
            current_graph,
            new_graph,
            0,
            operation,
        ):
            q.put(new_graph, remaining_ops)


def queue_two_qubit_operations(
    meta_graph: nx.Graph,
    graph_index: dict,
    q: queue.Queue,
    current_graph: nx.Graph,
):
    """
    Queue the two-qubit operations for processing.

    Args:
        meta_graph: The meta graph to which the new graphs will be added.
        graph_index: A dictionary mapping graphs to their indices in the meta graph.
        q: The queue to which the new graphs will be added.
        current_graph: The current graph being processed.
    """
    connected_components = list(nx.connected_components(current_graph))

    if connected_components == 1:
        return  # No two-qubit operations to apply

    for i in range(len(connected_components)):
        for j in range(i + 1, len(connected_components)):
            for node1 in connected_components[i]:
                for node2 in connected_components[j]:
                    if node1 % 2 == node2 % 2:
                        operation1 = f"CZ({node1},{node2})"
                        operation2 = f"F({node1},{node2})"
                        for operation in [operation1, operation2]:
                            queue_operation(
                                meta_graph,
                                graph_index,
                                q,
                                current_graph,
                                operation,
                                [],
                            )


def create_graph(n: int) -> tuple[nx.Graph, dict]:
    """
    Create the graph consisting of all the paths and nodes for shortest distance trial.
    """
    q = queue.Queue()
    graph_index = {}
    G = graph_utilities.gen_bell_tree(n)
    meta_graph = nx.Graph()

    q.put(G)  # Graph , gates,
    add_graph_to_meta(meta_graph, G, graph_index)

    while not q.empty():
        current_graph = q.get()

        queue_two_qubit_operations(meta_graph, graph_index, q, current_graph)

        # All local complementations
        for i in range(n):
            new_graph = graph_utilities.local_complement(current_graph, i)
            if not is_edge_present(
                meta_graph, graph_index, current_graph, new_graph, 0, f"LC({i})"
            ):
                q.put(new_graph)

    return tuple([meta_graph, graph_index])


if __name__ == "__main__":
    n = 4  # Number of qubits
    source_node = 0
    target_graph = nx.Graph()
    target_graph.add_nodes_from(range(n))
    target_graph.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])

    meta_graph, graph_index = create_graph(n)

    path = nx.shortest_path(
        meta_graph,
        source=source_node,
        target=get_node_index(target_graph, graph_index),
        weight="weight",
    )

    visualizer.run_dashboard(meta_graph)
