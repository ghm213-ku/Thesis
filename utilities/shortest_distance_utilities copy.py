from itertools import count

import networkx as nx
from typing import List, Tuple
import queue


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


def is_path_valid(nodes: int, path: list) -> bool:
    """
    Check if the given path is valid in the graph.

    Args:
        graph: The graph in which to check the path.
        path: A list of nodes representing a potential path in the graph.
    Returns:
        True if the path is valid, False otherwise.
    """
    G = nx.Graph()
    for edge in path:
        for j in range(nodes // 2):
            G.add_edge(2 * j, 2 * j + 1)  # Add edges for each qubit pair
        G.add_edge(edge[0], edge[1])

    try:
        nx.find_cycle(G, orientation="ignore")
        return False  # Cycle found, path is invalid
    except nx.NetworkXNoCycle:
        return True  # No cycle found, path is valid

    return True


def is_op_sequence_valid(nodes: int, operations: list) -> bool:
    """
    Check if the given sequence of operations is valid in the graph.

    Args:
        nodes: The number of nodes in the graph.
        operations: A list of operation strings representing a potential sequence of operations in the graph.
    Returns:
        True if the sequence of operations is valid, False otherwise.
    """
    path = []
    for operation in operations:
        op = operation.get("operation", "")
        if op.startswith(("LC")):
            continue  # Skip LC operations for path validation
        index_tuple = get_indices_from_operation(op)
        path += [index_tuple]  # Add the indices as a tuple to the path

    return is_path_valid(nodes, path)


def get_highlighted_items_from_path(path: list) -> tuple[list, list]:
    """
    Extracts the unique nodes from a given path.

    Args:
        path: A list of steps representing the path.

    Returns:
        A list of unique nodes in the path.
    """
    nodes = set()
    edges = []
    for step in path:
        nodes.add(step.get("source"))
        nodes.add(step.get("target"))
        edges.append((step.get("source"), step.get("target"), step.get("edge_key")))
    return list(nodes), edges


def get_signature(graph: nx.Graph, is_isomorphic: bool = False) -> tuple:
    """
    Get a unique signature for the graph based on its nodes and edges.

    Args:
        graph: The graph for which to get the signature.
        is_isomorphic: If True, use the isomorphic signature; otherwise, use the standard signature.
    Returns:
        A tuple containing sorted nodes and sorted edges of the graph.
    """
    if is_isomorphic:
        return nx.weisfeiler_lehman_graph_hash(graph)
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


def print_shortest_path(path: list):
    """
    Print the shortest path in a readable format.

    Args:
        path: A list of steps representing the shortest path.
    """
    if not path:
        print(f"No valid path exists .")
        return []
    for step in path:
        print(f"Step: {step.get('source')} -> {step.get('target')}")
        print(f"  Operation : {step.get('operation')})")
        print(f"  Cost      : {step.get('weight')}")
    total_cost = sum(step.get("weight") for step in path)
    print(f"Total Minimum Cost: {total_cost}")


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
    node_queue.put(
        (0, next(counter), 0, [])
    )  # weight, counter, current_node, path_so_far

    visited = set({0})

    while not node_queue.empty():
        cumulative_weight, _, current_node, path_so_far = node_queue.get()
        print(f"Visiting node: {current_node}, Cumulative weight: {cumulative_weight}")

        if nx.is_isomorphic(meta_graph.nodes[current_node]["graph"], target_graph):
            # if sorted(meta_graph.nodes[current_node]["graph"].edges()) == sorted(
            #     target_graph.edges()
            # ):
            print(f"Target graph found at node: {current_node}")
            return path_so_far

        for neighbor in meta_graph.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                weight = meta_graph[current_node][neighbor][
                    "weight"
                ]  # Default weight to 1 if not present
                operation = meta_graph[current_node][neighbor]["operation"]
                new_path = path_so_far + [
                    {
                        "source": current_node,
                        "target": neighbor,
                        "operation": operation,
                        "weight": weight,
                    }
                ]
                if is_op_sequence_valid(meta_graph.number_of_nodes(), new_path):
                    node_queue.put(
                        (
                            cumulative_weight + weight,
                            next(counter),
                            neighbor,
                            new_path,
                        )
                    )

    return []  # Return the path found, even if it doesn't reach the target
