import json
import networkx as nx

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
sys.path.append(str(Path(__file__).resolve().parent.parent / "utilities"))

import graph_utilities


def create_graphs_from_json(nodes: int):
    script_dir = Path(__file__).parent
    file_path = script_dir / f"{nodes}_nodes.json"
    graphs = []
    data = json.load(open(file_path, "r"))

    for key in data.keys():
        parsed_key = json.loads(key)
        edges = parsed_key.get("edges", [])

        # Create a new graph for this specific entry
        G = nx.Graph()
        G.add_edges_from(edges)
        graphs.append(G)

    return graphs


def create_recipes_from_json(nodes: int):
    script_dir = Path(__file__).parent
    file_path = script_dir / f"{nodes}_nodes.json"
    recipes = []
    data = json.load(open(file_path, "r"))

    for values in data.values():
        recipe = values.get("strings", [])
        recipes.append(recipe)

    return recipes


def get_graphs_from_recipes(nodes: int):
    recipes = create_recipes_from_json(nodes)
    graphs = []
    for recipe in recipes:
        G = graph_utilities.gen_bell_tree(nodes)
        for op in recipe:
            parts = op.split("(")
            if parts[0] == "CZ":
                G = graph_utilities.cz_gate_toggle(
                    G, int(parts[1].split(",")[0]), int(parts[1].split(",")[1][:-1])
                )
            elif parts[0] == "FUSION":
                G = graph_utilities.f_gate(
                    G, int(parts[1].split(",")[0]), int(parts[1].split(",")[1][:-1])
                )
            elif parts[0] == "LC":
                G = graph_utilities.local_complement(
                    G, int(parts[1].split(",")[0][:-1])
                )
        print(f"Graph from recipe: {recipe}")
        print(f"Graph edges: {G.edges()}")
        graphs.append(G)
    return graphs
