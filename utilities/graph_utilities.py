import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple
import graphstate_opt as gso
import random


def local_complement(G: nx.Graph, vertex: int) -> nx.Graph:
    """
    Perform local complementation on a graph at a specific vertex.

    Local complementation flips the edges among neighbors of the vertex.
    Equivalent to Mathematica's LocalComplement[G_, a_]

    Args:
        G: networkx graph
        vertex: Vertex to perform local complementation on

    Returns:
        New graph with local complementation applied
    """
    G_copy = G.copy()
    neighbors = list(G_copy.neighbors(vertex))

    for i, u in enumerate(neighbors):
        for v in neighbors[i + 1 :]:
            if G_copy.has_edge(u, v):
                G_copy.remove_edge(u, v)
            else:
                G_copy.add_edge(u, v)

    return G_copy


def delete_graphs_under_LC(
    graph_dict: dict[str, tuple[nx.Graph, int, List[str]]],
) -> dict[str, tuple[nx.Graph, int, List[str]]]:
    """
    Remove duplicates from a list of graphs which correspond to the same local complementation class.

    Args:
        graph_dict: Dictionary mapping graph strings to (graph, probability, strings) tuples

    Returns:
        Dictionary of graphs with isomorphic duplicates removed
    """
    unique_graphs = {}

    for key, (G, prob, strings) in graph_dict.items():
        is_duplicate = False
        G = find_mer(G, True)  # Get the minimum edge representative for the graph
        for H in unique_graphs:
            if nx.is_isomorphic(G, H):
                is_duplicate = True
                break

        if not is_duplicate:
            unique_graphs[key] = (G, prob, strings)

    return unique_graphs


def cz_gate_toggle(G: nx.Graph, i: int, j: int) -> nx.Graph:
    """
    CZ gate: toggle edge between two vertices.

    Args:
        G: networkx graph
        i, j: Vertex indices

    Returns:
        New graph with toggled edge
    """
    G_copy = G.copy()

    if G_copy.has_edge(i, j):
        G_copy.remove_edge(i, j)
    else:
        G_copy.add_edge(i, j)

    return G_copy


def f_gate(G: nx.Graph, i: int, j: int) -> nx.Graph:
    """
    Fusion/F-gate operation: complex graph update.

    Args:
        G: networkx graph
        i, j: Vertex indices

    Returns:
        New graph with F-gate applied
    """
    G_copy = G.copy()
    neighbors_j = list(G_copy.neighbors(j))

    for k in neighbors_j:
        G_copy.remove_edge(j, k)
        if G_copy.has_edge(i, k):
            G_copy.remove_edge(i, k)
        elif k != i:
            G_copy.add_edge(i, k)


    G_copy.add_edge(i, j)

    return G_copy


def gen_bell_tree(n: int) -> nx.Graph:
    """
    Generate tree for Bell pair postselection (even n).

    Args:
        n: Number of qubits (must be even)

    Returns:
        Graph representing the Bell pair tree
    """
    if n % 2 != 0:
        raise ValueError("n must be even")

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(1, n // 2 + 1):
        q1 = 2 * i - 2
        q2 = 2 * i - 1
        G.add_edge(q1, q2)

    return G


def gen_random_tree(n: int) -> nx.Graph:
    """
    Generate a random tree with n vertices and n-1 edges.

    Args:
        n: Number of vertices in the tree

    Returns:
        Random tree graph with n vertices and n-1 edges
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return nx.Graph()

    return nx.random_labeled_tree(n)


def generate_parity_constrained_tree(n):
    """
    Generates a tree with 2n nodes based on specific parity constraints.
    - Nodes 0 to 2n-1 are created. Evens and odds are pre-connected in pairs.
    - n-1 additional edges are added strictly between nodes of the same parity.
    """
    G = nx.Graph(directed=False)
    G.add_nodes_from(range(2 * n))

    # 2. Generate all valid candidate edges (strictly same parity)
    candidate_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            # Even-to-even candidate
            candidate_edges.append((2 * i, 2 * j))
            # Odd-to-odd candidate
            candidate_edges.append((2 * i + 1, 2 * j + 1))

    # Shuffle to randomize the tree generation
    random.shuffle(candidate_edges)

    # 3. Disjoint Set Union (DSU) setup
    parent = {node: node for node in G.nodes()}

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])  # Path compression
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False

    # Pre-merge the DSU components for the initial even-odd pairs
    for i in range(n):
        union(2 * i, 2 * i + 1)

    # 4. Build the tree by adding n-1 valid edges
    edges_added = 0
    for u, v in candidate_edges:
        # If the union is successful, no cycle is formed
        if union(u, v):
            G.add_edge(u, v, edge_type="added")
            edges_added += 1

            # Stop once we have exactly 2n - 1 edges total
            # (n initial edges + n - 1 new edges)
            if edges_added == n - 1:
                break

    return G


def get_graph_key(G: nx.Graph) -> str:
    """
    Returns a unique string hash based on the graph's topology.
    """
    return nx.weisfeiler_lehman_graph_hash(G)


def find_mer(G: nx.Graph, extensive: bool = False) -> nx.Graph:
    """
    Find the minimum edge representative (MER) of a graph.

    Args:
        G: Input graph
        extensive: Whether to use the extensive search method
    Returns:
        Graph with the minimum number of edges in its isomorphism class
    """
    # Generate all isomorphic graphs and find the one with the fewest edges
    if extensive:
        mer_graph = gso.edm_sa_ilp(G, 10000, 100)[0]
    else:
        mer_graph = gso.edm_sa_ilp(G, 100, 10)[0]
    # mer_graph.remove_edges_from(list(nx.selfloop_edges(mer_graph)))
    return mer_graph


def draw_graph(graph: nx.Graph, title: str = "graph"):
    plt.figure()
    plt.title(title)
    nx.draw_networkx(graph)
    plt.draw()
    plt.show(block=True)


def draw_graph_with_node_labels(G: nx.Graph, title: str = "graph"):
    weight_labels = {n: f"{n}:{G.nodes[n].get('weight', 'N/A')}" for n in G.nodes()}

    plt.figure()
    plt.title(title)
    nx.draw_networkx(
        G,
        labels=weight_labels,
    )
    plt.draw()
    plt.show(block=True)


if __name__ == "__main__":
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3])
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    f = f_gate(g, 2, 0)
    draw_graph(f, title="F-gate applied to graph")
