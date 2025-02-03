from jax import random, numpy as jnp
import jax
from jax import numpy as jnp, random, tree_map
from stadion.models import LinearSDE
from pprint import pprint
import networkx as nx

import matplotlib.pyplot as plt

from stadion.parameters import InterventionParameters
from stadion.notreks import notreks_loss, no_treks
from stadion.metrics import calculate_distances

from itertools import combinations

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
    
    # Step 3: Find pairs of nodes with no treks
    no_trek_nodes = [
        (u, v)
        for u, v in combinations(graph.nodes, 2)
        if not trek_graph.has_edge(u, v)
    ]
    
    return trek_graph, no_trek_nodes

def generate_acyclic_graph(d, sparsity, key):
    """
    Generates a random directed acyclic graph (DAG).

    Args:
        d (int): Number of variables (nodes).
        sparsity (float): Fraction of possible edges to include (0 to 1).
        key (jax.random.PRNGKey): Random key for reproducibility.
        seed (int, optional): Random seed for key initialization.

    Returns:
        networkx.DiGraph: A directed acyclic graph.
    """
    if not 0 <= sparsity <= 1:
        raise ValueError("Sparsity must be between 0 and 1.")

    dag = nx.DiGraph()

    # Add nodes
    dag.add_nodes_from(range(d))

    # Add edges while ensuring acyclicity
    for i in range(d):
        for j in range(i + 1, d):  # Only consider edges from lower to higher indices
            # Split the key to get a new subkey for each iteration
            key, subkey = jax.random.split(key)
            random_value = jax.random.uniform(subkey, minval=-3., maxval=3.)  # Random value in [0, 1)

            if random_value < sparsity:
                key, weight_key = jax.random.split(key)
                weight = jax.random.uniform(weight_key, minval=-1.0, maxval=1.0)  # Random weight between -1 and 1
                dag.add_edge(i, j, weight=weight)

    return dag


def sample_scm(dag, n, key, noise_dist="gaussian", noise_params=None, shift_intv = None):
    """
    Samples from a Structural Causal Model (SCM) defined by a DAG.

    Args:
        dag (networkx.DiGraph): A directed acyclic graph defining variable dependencies.
        n (int): Number of samples to generate.
        key (jax.random.PRNGKey): JAX random key for sampling.
        noise_dist (str): Type of noise ("gaussian" or "custom").
        noise_params (dict, optional): Parameters for the noise distribution.

    Returns:
        jnp.ndarray: Samples of shape (n, d) where d is the number of nodes in the DAG.
    """
    # Validate DAG
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("The input graph must be a directed acyclic graph (DAG).")
    
    # Topological sort of nodes for sampling in the correct order
    sorted_nodes = list(nx.topological_sort(dag))
    d = len(sorted_nodes)

    # Initialize data matrix
    samples = jnp.zeros((n, d))

    # Default noise parameters for Gaussian
    if noise_dist == "gaussian" and noise_params is None:
        noise_params = {"mean": 0.0, "std": 1.0}

    # Generate samples
    for idx, node in enumerate(sorted_nodes):
        parents = list(dag.predecessors(node))

        # Compute the value of the node based on its parents
        if parents:
            parent_indices = [sorted_nodes.index(p) for p in parents]
            weights = [dag.edges[(p, node)]["weight"] for p in parents]
            contribution = jnp.sum(samples[:, parent_indices] * jnp.array(weights), axis=1)
        else:
            contribution = 0.0

        # Add noise
        if noise_dist == "gaussian":
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=(n,)) * noise_params["std"] + noise_params["mean"]
        elif noise_dist == "custom" and callable(noise_params.get("sample")):
            noise = noise_params["sample"](key, n)
        else:
            raise ValueError("Unsupported noise distribution or missing parameters.")
        shift_intv_ = 0 if shift_intv == None else shift_intv[idx]
        samples = samples.at[:, idx].set(contribution + noise + shift_intv_)

    return samples

def plot_dag(graph):
    """
    Plots a networkx graph.

    Args:
        graph (networkx.DiGraph): The graph.
    """
    pos = nx.spring_layout(graph)  # You can also use other layouts like circular_layout
    plt.figure(figsize=(8, 6))  # Adjust the size as needed
    nx.draw(graph, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
    plt.title("Directed Acyclic Graph (DAG)")
    plt.show()


if __name__ == "__main__":
    # key = 1, d = 10
    key = random.PRNGKey(12)
    n, d = 1000, 10

    # # generate a dataset
    # key, subk = random.split(key)
    # w = random.normal(subk, shape=(d, d))

    # key, subk = random.split(key)
    # data = random.normal(subk, shape=(n, d)) @ w

    # sample two more datasets with shift interventions
    a, targets_a =  3, jnp.zeros(d).at[1].set(1)
    b, targets_b = -5, jnp.zeros(d).at[2].set(1)
    c, targets_c = -20, jnp.zeros(d).at[4].set(1)

    # key, subk_0, subk_1 = random.split(key, 3)
    # data_a = (random.normal(subk_0, shape=(n, d)) + a * targets_a) @ w
    # data_b = (random.normal(subk_1, shape=(n, d)) + b * targets_b) @ w
    
    # Generate a random DAG
    sparsity = 0.1
    
    no_trek_nodes = []
    while len(no_trek_nodes) == 0:
        key, subk = random.split(key)
        dag = generate_acyclic_graph(d, sparsity, subk)
        trek_graph, no_trek_nodes = build_trek_graph(dag)
    
    print(no_trek_nodes)
    
    plot_dag(dag)
    
    key, subk = random.split(key)
    data = sample_scm(dag, n, subk, noise_dist="gaussian")
    key, subk = random.split(key)
    data_a = sample_scm(dag, n, key, noise_dist="gaussian", shift_intv = a * targets_a)
    key, subk = random.split(key)
    data_b = sample_scm(dag, n, key, noise_dist="gaussian", shift_intv = b * targets_b)
    key, subk = random.split(key)
    data_c = sample_scm(dag, n, key, noise_dist="gaussian", shift_intv = c * targets_c)
    key, subk = random.split(key)
    data_c2 = sample_scm(dag, n, key, noise_dist="gaussian", shift_intv = c * targets_c)
    
    marg_indeps = jnp.array([no_trek_nodes, no_trek_nodes, no_trek_nodes])

    # fit stationary diffusion model
    model = LinearSDE(
            dependency_regularizer="NO TREKS", # Non-Structural",# "both", # 
            no_neighbors=True
        )
    key, subk = random.split(key)
    model.fit(
        subk,
        [data, data_a, data_b],
        targets=[jnp.zeros(d), targets_a, targets_b],
        steps=2*10000,
        marg_indeps=marg_indeps,
        dep=1
    )

    # get inferred model and intervention parameters
    param = model.param
    intv_param = model.intv_param

    # pprint(param)
    # pprint(intv_param)
    
    # in distribution test
    intv_param_a = intv_param.index_at(1)
    x_pred_a = model.sample(subk, 1000, intv_param=intv_param_a)

    distances_a = calculate_distances(data_a, x_pred_a)
    print("Mean Squared Error of the Means:", distances_a["mse_means"])
    print("Wasserstein Distance:", distances_a["wasserstein_distance"])
    
    # out of distribution test
    param_c = {
        "shift": c * targets_c, # intv_param.index_at(3)["shift"], # 
        "log_scale": jnp.zeros(d),
    }
    intv_param_c = InterventionParameters(parameters=param_c, targets=targets_c)
    
    x_pred_c = model.sample(subk, 1000, intv_param=intv_param_c)

    distances_c = calculate_distances(data_c, x_pred_c)
    print("Mean Squared Error of the Means:", distances_c["mse_means"])
    print("Wasserstein Distance:", distances_c["wasserstein_distance"])
    

    # distances_c2 = calculate_distances(data_c, data_c2)
    # print("Mean Squared Error of the Means:", distances_c2["mse_means"])
    # print("Wasserstein Distance:", distances_c2["wasserstein_distance"])
