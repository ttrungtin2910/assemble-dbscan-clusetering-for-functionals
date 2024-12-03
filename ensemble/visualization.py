import networkx as nx
import numpy as np
import matplotlib.axes

def visualize_matrix_as_graph_with_coordinates(
        G: nx.Graph,
        list_points: np.ndarray,
        index_cluster: int = 0,
        ax: matplotlib.axes.Axes = None,
    ):
    """
    Visualizes a graph from a similarity matrix with given point coordinates 
    and cluster index, and plots it on a specified matplotlib axis.

    Parameters
    ----------
    matrix : np.ndarray
        A 2D array representing the similarity matrix of the graph. Non-zero 
        entries indicate edges between nodes, with values representing edge weights.
    list_points : list or np.ndarray
        A list or 2D array of shape (n_points, 2), where each row contains the 
        x and y coordinates of a point. These coordinates are used to position 
        nodes in the graph.
    index_cluster : int
        The index or label of the cluster being visualized. This is used in 
        the subplot title to identify the cluster.
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib subplot axis on which to draw the graph.

    Notes
    -----
    - Nodes are positioned using the `list_points` coordinates.
    - Edges are drawn only for non-zero weights in the similarity matrix.
    - Nodes are colored based on connectivity:
        - Green for nodes connected to at least one edge.
        - Red for isolated nodes (nodes with no edges).
    - Edge transparency (alpha) is scaled based on the weight of the edge.
    - The title of the plot indicates the cluster index, e.g., "Graph Visualization Cluster #1".
    - Axes are turned off for a cleaner visualization.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)
    >>> visualize_matrix_as_graph_with_coordinates(matrix, points, 1, ax)
    >>> plt.show()
    """
    # Create positions for nodes based on given list_points
    positions = {i: tuple(list_points[i]) for i in range(len(list_points))}

    # Determine node colors based on connectivity
    node_colors = []
    for node in G.nodes:
        # Green for connected nodes, red for isolated nodes
        if any((node in edge) for edge in G.edges):
            node_colors.append('green')
        else:
            node_colors.append('red')

    # Extract edge weights to adjust edge transparency (alpha)
    weights = nx.get_edge_attributes(G, 'weight')
    max_weight = max(weights.values()) if weights else 1  # Prevent division by zero

    # Scale edge colors with alpha based on weight
    edge_colors = [(0, 0, 0, weight / max_weight) for _, weight in weights.items()]  # RGBA format

    # Draw nodes with the specified colors
    nx.draw_networkx_nodes(G, positions, node_color=node_colors, node_size=30, ax=ax)

    # Draw edges with transparency based on weight
    nx.draw_networkx_edges(G, positions, edge_color=edge_colors, alpha=0.01, ax=ax)

    # Set the title for this subplot
    ax.set_title(f"Graph Visualization Cluster #{int(index_cluster)}")
    ax.axis('off')  # Turn off axis display