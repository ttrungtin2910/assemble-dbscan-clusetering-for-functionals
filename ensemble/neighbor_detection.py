import numpy as np
import networkx as nx
from cdlib import algorithms
import matplotlib.pyplot as plt

def create_graph(
        matrix: np.ndarray
    ):
    
    n = len(matrix)  # Number of nodes

    # Initialize an empty graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(range(n))

    # Add edges with weights from the matrix
    for i in range(n):
        for j in range(i + 1, n):
            weight = matrix[i][j]
            # Add edges only if the weight is non-zero
            if weight != 0:
                G.add_edge(i, j, weight=weight)
    
    return G

def leiden_alg(G, list_points, output_filename):
    positions = {i: tuple(list_points[i]) for i in range(len(list_points))}

    # Detect communities using the Leiden algorithm
    # coms = algorithms.leiden(G)
    coms = algorithms.louvain(G)

    # Get the list of communities and assign a different color for each community
    community_colors = coms.communities
    color_map = {}

    list_labels = [None] * len(positions)

    # Assign colors to each node based on the community it belongs to
    for i, comm in enumerate(community_colors):
        for node in comm:
            color_map[node] = i
            list_labels[node] = i

    # Draw the graph
    plt.figure(figsize=(8, 8))

    # Draw nodes with colors corresponding to their community
    node_color = [color_map[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, positions, node_size=20, cmap=plt.cm.rainbow, node_color=node_color)

    # Optionally, draw edges and labels (you can uncomment if needed)
    # nx.draw_networkx_edges(G, positions, alpha=0.5)
    # nx.draw_networkx_labels(G, positions, font_size=12)

    # Create a legend for the communities
    unique_communities = list(set(node_color))
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.rainbow(c / len(unique_communities)), markersize=10) for c in unique_communities]
    labels = [f"Community {c}" for c in unique_communities]
    plt.legend(handles, labels, title="Communities")

    # Set the title of the plot
    plt.title('Communities in Graph (Leiden Algorithm)')

    # Adjust layout to ensure everything fits
    plt.tight_layout()

    # Save the plot to the specified output file
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    return list_labels