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
    coms = algorithms.leiden(G)

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
    # pos = nx.spring_layout(G)  # Generate layout for the graph
    plt.figure(figsize=(8, 8))

    # Draw nodes with colors corresponding to their community
    nx.draw_networkx_nodes(G, positions, node_size=20, cmap=plt.cm.rainbow, node_color=[color_map[node] for node in G.nodes()])
    # nx.draw_networkx_edges(G, positions, alpha=0.5)
    # nx.draw_networkx_labels(G, positions, font_size=12)

    # Set the title of the plot
    plt.title('Communities in Graph (Leiden Algorithm)')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    return list_labels
