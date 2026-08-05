import dash
from dash import html, Input, Output
import dash_cytoscape as cyto


def _nx_to_cyto_elements(
    nx_g, is_metagraph=False, highlight_nodes=None, highlight_edges=None
):
    if highlight_nodes is None:
        highlight_nodes = []
    if highlight_edges is None:
        highlight_edges = []

    highlight_nodes = [str(n) for n in highlight_nodes]

    # Pre-process edge tuples into string IDs (handles both regular and multi-graphs)
    highlight_edge_ids = set()
    for edge in highlight_edges:
        if len(edge) == 3:  # MultiDiGraph: (u, v, key)
            highlight_edge_ids.add(f"{edge[0]}-{edge[1]}-{edge[2]}")
        elif len(edge) == 2:  # DiGraph: (u, v)
            highlight_edge_ids.add(f"{edge[0]}-{edge[1]}")

    elements = []

    # 1. Process Nodes
    for node, data in nx_g.nodes(data=True):
        node_dict = {"data": {"id": str(node), "label": str(node)}}
        if is_metagraph and str(node) in highlight_nodes:
            node_dict["classes"] = "highlighted-node"
        elements.append(node_dict)

    # 2. Process Edges
    is_multi = nx_g.is_multigraph()
    edges = nx_g.edges(data=True, keys=True) if is_multi else nx_g.edges(data=True)

    for edge in edges:
        if is_multi:
            u, v, key, data = edge
            edge_id = f"{u}-{v}-{key}"
        else:
            u, v, data = edge
            edge_id = f"{u}-{v}"

        edge_data = {"source": str(u), "target": str(v), "id": edge_id}

        if is_metagraph:
            edge_data["label"] = (
                f"{data.get('operation', '')} (w={data.get('weight', '')})"
            )

        edge_dict = {"data": edge_data}

        # Tag the edge if it's in our highlight list
        if is_metagraph and edge_id in highlight_edge_ids:
            edge_dict["classes"] = "highlighted-edge"

        elements.append(edge_dict)

    return elements


# Add highlight_nodes as an optional parameter
def run_dashboard(meta_graph, highlight_nodes=None, highlight_edges=None, port=8050):
    app = dash.Dash(__name__)

    app.layout = html.Div(
        style={
            "display": "flex",
            "flexDirection": "row",
            "height": "100vh",
            "fontFamily": "sans-serif",
        },
        children=[
            html.Div(
                style={
                    "width": "60%",
                    "borderRight": "2px solid #ccc",
                    "padding": "10px",
                },
                children=[
                    html.H2("Metagraph"),
                    cyto.Cytoscape(
                        id="meta-graph-view",
                        # Pass both highlight lists to the helper:
                        elements=_nx_to_cyto_elements(
                            meta_graph,
                            is_metagraph=True,
                            highlight_nodes=highlight_nodes,
                            highlight_edges=highlight_edges,
                        ),
                        layout={
                            "name": "cose",
                            "nodeRepulsion": 400000,
                            "idealEdgeLength": 150,
                            "nodeOverlap": 10,
                            "padding": 30,
                        },
                        style={"width": "100%", "height": "800px"},
                        stylesheet=[
                            # --- Nodes ---
                            {
                                "selector": "node",
                                "style": {
                                    "content": "data(label)",
                                    "background-color": "#0074D9",
                                    "text-valign": "bottom",
                                    "text-halign": "center",
                                    "text-margin-y": "5px",
                                },
                            },
                            {
                                "selector": ".highlighted-node",
                                "style": {
                                    "background-color": "#FF4136",
                                    "border-color": "#85144b",
                                    "border-width": 3,
                                    "color": "#FF4136",
                                    "font-weight": "bold",
                                },
                            },
                            # --- Edges ---
                            {
                                "selector": "edge",
                                "style": {
                                    "content": "data(label)",
                                    "font-size": "11px",
                                    "curve-style": "bezier",
                                    "control-point-step-size": 70,
                                    "target-arrow-shape": "triangle",
                                    "target-arrow-color": "#999999",
                                    "line-color": "#999999",
                                    "text-background-opacity": 1,
                                    "text-background-color": "#ffffff",
                                    "text-background-padding": "4px",
                                    "text-background-shape": "roundrectangle",
                                    "text-border-color": "#dddddd",
                                    "text-border-width": 1,
                                    "color": "#333333",
                                },
                            },
                            # HIGHLIGHTED EDGE STYLE
                            {
                                "selector": ".highlighted-edge",
                                "style": {
                                    "line-color": "#FF4136",  # Red line
                                    "target-arrow-color": "#FF4136",  # Red arrow
                                    "width": 4,  # Thicker line
                                    "text-border-color": "#FF4136",  # Red border around the label box
                                    "text-border-width": 2,
                                    "color": "#FF4136",  # Red text
                                    "font-weight": "bold",
                                    "z-index": 999,  # Bring highlighted edge to the front
                                },
                            },
                        ],
                    ),
                ],
            ),
            # RIGHT PANEL (Unchanged)
            html.Div(
                style={"width": "40%", "padding": "10px"},
                children=[
                    html.H2(
                        id="inner-graph-title", children="Inner Graph: (None Selected)"
                    ),
                    cyto.Cytoscape(
                        id="inner-graph-view",
                        elements=[],
                        layout={"name": "circle"},
                        style={"width": "100%", "height": "600px"},
                        stylesheet=[
                            {
                                "selector": "node",
                                "style": {
                                    "content": "data(label)",
                                    "background-color": "#FF851B",
                                },
                            },
                            {
                                "selector": "edge",
                                "style": {
                                    "curve-style": "bezier",
                                    "target-arrow-shape": "triangle",
                                    "target-arrow-color": "#999999",
                                    "line-color": "#999999",
                                },
                            },
                        ],
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        [
            Output("inner-graph-view", "elements"),
            Output("inner-graph-title", "children"),
        ],
        [Input("meta-graph-view", "tapNodeData")],
    )
    def display_inner_graph(node_data):
        if not node_data:
            return [], "Inner Graph: (None Selected)"

        clicked_node_id = node_data["id"]

        # 1. Handle Type Mismatch:
        # If the string ID isn't in the graph, check if it was originally an integer
        if clicked_node_id not in meta_graph.nodes:
            try:
                clicked_node_id = int(clicked_node_id)
            except ValueError:
                pass  # It's just a string that actually isn't there

        # 2. Check if node actually exists after type correction
        if clicked_node_id not in meta_graph.nodes:
            return [], f"ERROR: Node '{clicked_node_id}' not found in metagraph."

        node_attributes = meta_graph.nodes[clicked_node_id]

        # 3. Check if 'graph' attribute exists
        if "graph" not in node_attributes:
            return (
                [],
                f"ERROR: Node '{clicked_node_id}' is missing the 'graph' attribute.",
            )

        # 4. Success path
        inner_nx_graph = node_attributes["graph"]

        # If the inner graph exists but is empty, notify the user
        if len(inner_nx_graph.nodes()) == 0:
            return [], f"Inner Graph: {clicked_node_id} (Empty Graph)"

        inner_cyto_elements = _nx_to_cyto_elements(inner_nx_graph, is_metagraph=False)

        return inner_cyto_elements, f"Inner Graph: {clicked_node_id}"

    # Run the server
    app.run(debug=False, port=port)
