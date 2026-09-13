from sage.all import Graph
from sage.graphs.graph_decompositions.rankwidth import rank_decomposition
import networkx as nx

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
sys.path.append(str(Path(__file__).resolve().parent.parent / "mir_graphs"))
import mir_graphs

graphs = mir_graphs.create_graphs_from_json(mir_graphs.data)

for i, g in enumerate(graphs):
    rw, tree = rank_decomposition(Graph(g))
    print(f"The Rank-Width is: {rw}")
    # print(f"The Decomposition Tree is: {tree}")
    # plot = tree.plot(layout="spring", vertex_size=500, vertex_labels=True)
    # plot.save(f"decomposition_tree_{i}.png")
