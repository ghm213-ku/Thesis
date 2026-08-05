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


def get_indices_from_operation(operation: str) -> tuple[int, int]:
    """
    Extract the two indices from an operation string.

    Args:
        operation: A string representing the operation, e.g., "CZ(0,1)".

    Returns:
        A tuple containing the two indices as integers.
    """
    # Remove the operation name and parentheses, then split by comma
    indices_str = operation.split("(")[1].rstrip(")")
    try:
        i, j = map(int, indices_str.split(","))
    except ValueError:
        return -1, -1  # Return -1 for j if conversion fails
    return i, j


def is_path_valid(path: list) -> bool:
    """
    Check if the given path is valid in the graph.

    Args:
        graph: The graph in which to check the path.
        path: A list of nodes representing a potential path in the graph.
    Returns:
        True if the path is valid, False otherwise.
    """
    G = nx.Graph()
    for i in path:
        operation = i.get("operation")
        index_tuple = get_indices_from_operation(operation)
        if index_tuple != (-1, -1):
            G.add_edge(index_tuple[0], index_tuple[1])

    try:
        nx.find_cycle(G, orientation="ignore")
        return False  # Cycle found, path is invalid
    except nx.NetworkXNoCycle:
        return True  # No cycle found, path is valid

    return True


def shortest_path(meta_graph: nx.Graph, target_graph: nx.Graph):
    """
    Find the shortest path in a multi-graph from source_node to target_node using Dijkstra's algorithm.

    Args:
        meta_graph: The multi-graph in which to find the shortest path.
        target_graph: The target graph for the path.

    Returns:
        A list of nodes representing the shortest path from source_node to target_node.
    """
    node_queue = queue.PriorityQueue()
    counter = count()
    node_queue.put((0, next(counter), source_node, []))

    visited = set()

    while not node_queue.empty():
        cumulative_weight, _, current_node, path_so_far = node_queue.get()
        print(f"Visiting node: {current_node}, Cumulative weight: {cumulative_weight}")
        visited.add(current_node)

        if nx.is_isomorphic(meta_graph.nodes[current_node]["graph"], target_graph):
            return path_so_far

        for neighbor in meta_graph.neighbors(current_node):
            if neighbor in visited:
                continue

            edge_data = meta_graph.get_edge_data(current_node, neighbor)
            for _, edge in edge_data.items():
                if is_path_valid(path_so_far + [edge]):
                    edge_weight = edge.get("weight", 1)
                    node_queue.put(
                        (
                            cumulative_weight + edge_weight,
                            next(counter),
                            neighbor,
                            path_so_far
                            + [
                                {
                                    "source": current_node,
                                    "target": neighbor,
                                    "operation": edge.get("operation"),
                                    "weight": edge_weight,
                                }
                            ],
                        )
                    )

    return []  # Return the path found, even if it doesn't reach the target


def get_custom_multigraph_execution_path(multi_graph: nx.Graph, target_graph: nx.Graph):
    path = shortest_path(multi_graph, target_graph)
    if not path:
        print(f"No valid path exists .")
        return []
    for step in path:
        print(f"Step: {step.get('source')} -> {step.get('target')}")
        print(f"  Operation : {step.get('operation')})")
        print(f"  Cost      : {step.get('weight')}")
    total_cost = sum(step.get("weight") for step in path)
    print(f"Total Minimum Cost: {total_cost}")
    return path


def get_multigraph_execution_path(
    multi_graph, source_node, target_node, weight_attr="weight"
):
    try:
        # 1. Get the raw node path
        node_path = nx.dijkstra_path(
            multi_graph, source_node, target_node, weight=weight_attr
        )
    except nx.NetworkXNoPath:
        print(f"No valid path exists between {source_node} and {target_node}.")
        return []

    execution_sequence = []
    total_cost = 0

    print(f"--- Optimal Path from {source_node} to {target_node} ---")

    # 2. Iterate through the path step-by-step
    for i in range(len(node_path) - 1):
        u = node_path[i]
        v = node_path[i + 1]

        # In a MultiGraph, get_edge_data returns a dictionary of ALL edges between u and v
        # Format: { edge_key_0: {data}, edge_key_1: {data} }
        all_edges = multi_graph.get_edge_data(u, v)

        # 3. Find the specific edge that Dijkstra actually used (the one with the lowest weight)
        best_edge_key = None
        min_weight = float("inf")

        for key, edge_data in all_edges.items():
            # Get the weight (fallback to 1 if missing, which is NetworkX's default behavior)
            current_weight = edge_data.get(weight_attr, 1)

            if current_weight < min_weight:
                min_weight = current_weight
                best_edge_key = key

        # 4. Extract the exact operation and data
        best_edge_data = all_edges[best_edge_key]
        operation = best_edge_data.get("operation", "unknown_operation")

        print(f"Step {i+1}: {u} -> {v}")
        print(f"  Operation : {operation} (Edge Key: {best_edge_key})")
        print(f"  Cost      : {min_weight}")

        total_cost += min_weight
        execution_sequence.append(
            {
                "source": u,
                "target": v,
                "edge_key": best_edge_key,
                "operation": operation,
                "weight": min_weight,
                "full_data": best_edge_data,
            }
        )

    print(f"----------------------------------------")
    print(f"Total Minimum Cost: {total_cost}")

    return execution_sequence


def get_isomorphic_signature(G: nx.Graph) -> str:
    # This returns a string that is identical for any two isomorphic graphs
    return nx.weisfeiler_lehman_graph_hash(G)


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
    meta_graph = (
        nx.MultiDiGraph()
    )  # Use MultiDiGraph to allow multiple edges between nodes
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
    target_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

    meta_graph, graph_index = create_graph(n)

    get_custom_multigraph_execution_path(meta_graph, target_graph)

    visualizer.run_dashboard(meta_graph, port=8050)
