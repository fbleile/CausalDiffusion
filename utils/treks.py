# -*- coding: utf-8 -*-

from jax import numpy as jnp, random, tree_map
import networkx as nx

from itertools import combinations
from matplotlib import pyplot as plt

def build_trek_graph(graph):
    """
    Constructs the trek graph for a given AG.
    
    Args:
        graph (nx.DiGraph): A directed graph.
        
    Returns:
        trek_graph (nx.Graph): The trek graph (undirected).
        no_trek_nodes (list): List of pairs of nodes with no treks between them.
    """
    
    # Step 1: Compute ancestors for each node
    ancestors = {node: set(nx.ancestors(graph, node)) | {node} for node in graph.nodes}
    
    # Step 2: Build the trek graph
    trek_graph = nx.Graph()
    trek_graph.add_nodes_from(graph.nodes)
    
    for u, v in combinations(graph.nodes, 2):  # Iterate over all pairs of nodes
        if ancestors[u] & ancestors[v]:  # Check if they share a common ancestor
            trek_graph.add_edge(u, v)
    
    # pos = nx.spring_layout(trek_graph)  # You can also use other layouts like circular_layout
    # plt.figure(figsize=(8, 6))  # Adjust the size as needed
    # nx.draw(trek_graph, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
    # plt.show()
    
    return trek_graph

def get_all_missing_treks(mask):
    
    nx_graph = nx.from_numpy_array(mask, create_using=nx.DiGraph())
    trek_graph = build_trek_graph(nx_graph)
    
    # Get the complement (inverse) graph
    inv_trek_graph = nx.complement(trek_graph)
    
    # Get edge list as tuples
    edge_list = list(inv_trek_graph.edges)
    
    return edge_list
    
    