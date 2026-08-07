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

    if index1 == index2:
        return True  # Self loop

    edge_data = meta_graph.get_edge_data(index1, index2)
    if edge_data is not None:
        if isinstance(meta_graph, nx.MultiGraph):
            for _, attrs in edge_data.items():
                if attrs.get("operation") == operation:
                    return True  # Edge already exists with the same operation
        elif edge_data.get("operation") == operation:
            return True  # Edge already exists with the same operation

    meta_graph.add_edge(index1, index2, weight=weight, operation=operation)
    if operation.startswith("  ") or operation.startswith("CZ"):
        meta_graph.add_edge(index2, index1, weight=weight, operation=operation)

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
            q.put((new_graph, remaining_ops))
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
            q.put((new_graph, remaining_ops))
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
            q.put((new_graph, remaining_ops))


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

    for i, operation in enumerate(operations):
        remaining_ops = operations[:i] + operations[i + 1 :]
        for j in ["CZ", "F"]:
            op_str = f"{j}({operation[0]},{operation[1]})"
            queue_operation(
                meta_graph,
                graph_index,
                q,
                current_graph,
                op_str,
                remaining_ops,
            )
            queue_operation(
                meta_graph,
                graph_index,
                q,
                current_graph,
                f"{j}({operation[1]},{operation[0]})",
                remaining_ops,
            )


def create_graph(n: int) -> tuple[nx.Graph, dict]:
    """
    Create the graph consisting of all the paths and nodes for shortest distance trial.
    """
    q = queue.Queue()
    graph_index = {}
    G = graph_utilities.gen_bell_tree(n)
    meta_graph = (
        nx.MultiDiGraph()
    )  # Use MultiDiGraph to allow multiple edges between nodes
    operations = [[0, 2], [1, 3]]  # List of operations that can be applied

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
    target_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

    meta_graph, graph_index = create_graph(n)

    path = shortest_path(meta_graph, target_graph)
    print_shortest_path(path)

    highlighted_nodes, highlighted_edges = get_highlighted_items_from_path(path)

    visualizer.run_dashboard(
        meta_graph,
        highlight_nodes=highlighted_nodes,
        highlight_edges=highlighted_edges,
        port=8050,
    )
