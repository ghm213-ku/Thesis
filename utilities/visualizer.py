import dash
from dash import html, Input, Output
import dash_cytoscape as cyto


def _nx_to_cyto_elements(nx_g, is_metagraph=False, highlight_nodes=None):
    """Helper function to convert NetworkX to Cytoscape JSON."""
    if highlight_nodes is None:
        highlight_nodes = []
        
    # Convert highlight list to strings just in case they were passed as integers
    highlight_nodes = [str(n) for n in highlight_nodes]
    
    elements = []
    
    # 1. Process Nodes
    for node, data in nx_g.nodes(data=True):
        node_data = {'data': {'id': str(node), 'label': str(node)}}
        
        # If this node is in our highlight list, tag it with a CSS class
        if is_metagraph and str(node) in highlight_nodes:
            node_data['classes'] = 'highlighted-node'
            
        elements.append(node_data)
        
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
            
        edge_data = {
            'source': str(u), 
            'target': str(v), 
            'id': edge_id
        }
        
        if is_metagraph:
            edge_data['label'] = f"{data.get('operation', '')} (w={data.get('weight', '')})"
            
        elements.append({'data': edge_data})
        
    return elements


def run_dashboard(meta_graph, port=8050):
    """
    Initializes and runs the Dash web application using the provided metagraph.
    """
    app = dash.Dash(__name__)

    # Define the layout using the passed meta_graph
    app.layout = html.Div(
        style={
            "display": "flex",
            "flexDirection": "row",
            "height": "100vh",
            "fontFamily": "sans-serif",
        },
        children=[
            # LEFT PANEL
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
                        elements=_nx_to_cyto_elements(meta_graph, is_metagraph=True),
                        # 1. TUNE THE PHYSICS: Force nodes further apart
                        layout={
                            "name": "cose",
                            "nodeRepulsion": 900000,  # Push nodes far apart
                            "idealEdgeLength": 150,  # Make edges longer
                            "nodeOverlap": 10,
                            "padding": 30,
                        },
                        style={"width": "100%", "height": "800px"},
                        stylesheet=[
                            # 2. FIX NODE HITBOXES: Move labels below the nodes
                            {
                                "selector": "node",
                                "style": {
                                    "content": "data(label)",
                                    "background-color": "#0074D9",
                                    "text-valign": "bottom",  # Puts text below the circle
                                    "text-halign": "center",
                                    "text-margin-y": "5px",  # Adds a small gap
                                },
                            },
                            # 3. FIX EDGE READABILITY: Add a solid background to edge text
                            # 3. FIX EDGE READABILITY & ADD ARROWS
                            {
                                "selector": "edge",
                                "style": {
                                    "content": "data(label)",
                                    "font-size": "11px",
                                    "curve-style": "bezier",
                                    "control-point-step-size": 70,  # <--- ADDS SPACING BETWEEN MULTIPLE EDGES
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
                        ],
                    ),
                ],
            ),
            # RIGHT PANEL
            html.Div(
                style={"width": "40%", "padding": "10px"},
                children=[
                    html.H2(
                        id="inner-graph-title", children="Inner Graph: (None Selected)"
                    ),
                    cyto.Cytoscape(
                        id="inner-graph-view",
                        elements=[],
                        layout={
                            "name": "circle",
                        },
                        style={"width": "100%", "height": "600px"},
                        stylesheet=[
                            {
                                "selector": "node",
                                "style": {
                                    "content": "data(label)",
                                    "background-color": "#FF851B",
                                },
                            },
                            {"selector": "edge", "style": {"curve-style": "bezier"}},
                        ],
                    ),
                ],
            ),
        ],
    )

    # Define the callback inside the wrapper so it has access to `meta_graph`
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
