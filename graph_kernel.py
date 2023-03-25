import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from stabilizer_tools import StabilizerTableau, count_ones

def dist_from_graph(graph, num_layers=3, shots=100, exact=False):
    """
    Generate a probability distribution from a graph. This is done by acting 
    on the state |0>^n with a Clifford circuit based on the graph structure, 
    where n is the number of nodes of the graph. The number of 1s in the 
    resulting quantum state is measured. This is repeated a number of times 
    and the resulting distribution is returned.

    Args:
        graph: a networkx graph
        num_layers: the number of repetitions of the Clifford circuit
        shots: the number of computational basis measurements made.

    Returns:
        an numpy array of size (n + 1,) where the i-th entry represents the 
        probability of the final state having i 1s in it.
    """

    qubit_mapping = {}
    for i, node in enumerate(graph.nodes):
        qubit_mapping[node] = i

    num_qubits = graph.number_of_nodes()

    S = StabilizerTableau(num_qubits)
    D = StabilizerTableau(num_qubits, destabilizers=True)

    for _ in range(num_layers):
        for node in graph.nodes():
            q = qubit_mapping[node]
            S.conjugate('h', q)
            D.conjugate('h', q)
            S.conjugate('s', q)
            D.conjugate('s', q)
        for edge in graph.edges():
            source = qubit_mapping[edge[0]]
            target = qubit_mapping[edge[1]]
            S.conjugate('cx', source, target)
            D.conjugate('cx', source, target)
            S.conjugate('sdg', target)
            D.conjugate('sdg', target)
            S.conjugate('cx', source, target)
            D.conjugate('cx', source, target)
            S.conjugate('cz', source, target)
            D.conjugate('cz', source, target)
        # for node in graph.nodes():
        #     neighbors = list(graph.neighbors(node))
        #     for i in range(1, len(neighbors)):
        #         source = qubit_mapping[neighbors[i - 1]]
        #         target = qubit_mapping[neighbors[i]]
        #         S.conjugate('cx', source, target)
        #     q = qubit_mapping[neighbors[-1]]
        #     S.conjugate('s', q)
        #     for i in range(len(neighbors) - 1, 0, -1):
        #         source = qubit_mapping[neighbors[i - 1]]
        #         target = qubit_mapping[neighbors[i]]
        #         S.conjugate('cx', source, target)

    for node in graph.nodes():
            q = qubit_mapping[node]
            S.conjugate('h', q)
            D.conjugate('h', q)

    return count_ones(S, D, shots=shots, exact=exact)

# def dist_from_graph_test(graph, discrete_parameters, z_basis=True, shots=100, exact=False):

#     qubit_mapping = {}
#     for i, node in enumerate(graph.nodes):
#         qubit_mapping[node] = i

#     num_qubits = graph.number_of_nodes()

#     S = StabilizerTableau(num_qubits)
#     D = StabilizerTableau(num_qubits, destabilizers=True)


#     num_layers = discrete_parameters.shape[0]

#     for i in range(num_layers):
#         for node in graph.nodes():
        
#             q = qubit_mapping[node]
#             if discrete_parameters[i, 0] == 0: gate = 's'
#             else: gate = 'sdg'
#             S.conjugate('h', q)
#             D.conjugate('h', q)
#             S.conjugate(gate, q)
#             D.conjugate(gate, q)
#             # assert S.destabilizer_check(D)
        
#         for edge in graph.edges():
            
#             source = qubit_mapping[edge[0]]
#             target = qubit_mapping[edge[1]]

#             if discrete_parameters[i, 1] == 0: gate = 's'
#             else: gate = 'sdg'
#             S.conjugate('cx', source, target)
#             D.conjugate('cx', source, target)
#             # assert S.destabilizer_check(D)
#             S.conjugate(gate, target)
#             D.conjugate(gate, target)
#             # assert S.destabilizer_check(D)
#             S.conjugate('cx', source, target)
#             D.conjugate('cx', source, target)
#             # assert S.destabilizer_check(D)

#             if discrete_parameters[i, 2] == 0: gate = 'id'
#             else: gate = 'cz'
#             S.conjugate(gate, source, target)
#             D.conjugate(gate, source, target)
#             # assert S.destabilizer_check(D)
        
#             if discrete_parameters[i, 3] == 0: gate = 's'
#             else: gate = 'sdg'
#             S.conjugate(gate, source)
#             D.conjugate(gate, source)
#             # assert S.destabilizer_check(D)
#             S.conjugate(gate, target)
#             D.conjugate(gate, target)
#             # assert S.destabilizer_check(D)

#         # for node in graph.nodes():
#         #     neighbors = list(graph.neighbors(node))
#         #     for i in range(1, len(neighbors)):
#         #         source = qubit_mapping[neighbors[i - 1]]
#         #         target = qubit_mapping[neighbors[i]]
#         #         S.conjugate('cx', source, target)
#         #     q = qubit_mapping[neighbors[-1]]
#         #     S.conjugate('s', q)
#         #     for i in range(len(neighbors) - 1, 0, -1):
#         #         source = qubit_mapping[neighbors[i - 1]]
#         #         target = qubit_mapping[neighbors[i]]
#         #         S.conjugate('cx', source, target)

#     if z_basis:
#         for node in graph.nodes():
#             q = qubit_mapping[node]
#             S.conjugate('h', q)
#             D.conjugate('h', q)
#     # assert S.destabilizer_check(D)

#     # results = S.sample_z_basis(destabilizers=D, shots=shots)

#     # occupation_numbers = np.zeros(num_qubits + 1)
#     # for key, value in results.items():
#     #     occupation_numbers[key.count('1')] += value

#     return count_ones(S, D, shots=shots, exact=exact)

def shannon(dist):
    """
    Return the Shannon entropy of a distribution
    """
    return -np.sum(dist*np.log(dist, where=(dist!=0)))

def jensen_shannon(dist1, dist2):
    """
    Return the Jensen-Shannon divergence between two distributions
    """
    length = max(len(dist1), len(dist2))
    dist1 = np.pad(dist1, ((0, length - len(dist1))))
    dist2 = np.pad(dist2, ((0, length - len(dist2))))
    return shannon((dist1 + dist2)/2) - (shannon(dist1) + shannon(dist2))/2

def kernel_entry(dist1, dist2, coeff=40):
    """
    A kernel function between distributions
    """
    return np.exp(-coeff*jensen_shannon(dist1, dist2))

def kernel_function(X1, X2, coeff=40):
    """
    Calculate the kernel between distributions in X1 and X2

    Returns:
        a numpy array of size (X1.shape[0], X2.shape[0])
    """
    kernel = np.zeros((X1.shape[0], X2.shape[0]))
    for i, j in np.ndindex(kernel.shape):
        kernel[i, j] = kernel_entry(X1[i], X2[j], coeff=coeff)
    return kernel

def triangular_random_walk(num_nodes, p_acceptance, sublattice=None):
    """
    Make a graph from a random walk on a triangular lattice
    
    Args:
        num_nodes: the number of nodes for the graph to have
        p_acceptance: if a preferred sublattice is given, the probability 
            that an unpreferred vertex is accepted to the random walk if it is 
            reached
        sublattice: a preferred sublattice whose vertices are never rejected 
            for the random walk if they are encountered. Options are None 
            (all vertices are preferred), 'honeycomb' (a hexagonal sublattice 
            is preferred), and 'kagome' (a kagome sublattice preferred)
    
    Returns:
        a networkx graph with nodes labeled by their position in the lattice
    """

    if sublattice is None:
        is_favored = lambda x: True
    elif sublattice == 'honeycomb':
        is_favored = lambda x: (x[0] + x[1]) % 3 != 0
    elif sublattice == 'kagome':
        is_favored = lambda x: (x[0] % 2 != 0 or x[1] % 2 != 0)
    else:
        raise ValueError(f'sublattice {sublattice} is not implemented')

    graph = nx.Graph()
    graph.add_node((1, 1))

    walk = [np.array([1, 1])]
    head = np.array([1, 1])
    order_visited = {(1, 1): 0}
    node_count = 1
    vectors = np.array(
        [[1, 0],
         [1, 1],
         [0, 1],
         [-1, 0],
         [-1, -1],
         [0, -1]]
    )
    while node_count < num_nodes:
        next = head + vectors[np.random.randint(6)]
        if is_favored(next) or np.random.binomial(1, p_acceptance):
            # update graph
            graph.add_node(tuple(next))
            graph.add_edge(tuple(head), tuple(next))
            # update walk
            walk.append(next.copy())
            head = next.copy()
            # update node_count
            if order_visited.get(tuple(head)) is None:
                node_count += 1
                order_visited[tuple(head)] = node_count

    return graph

def triangular_lattice_draw(graph, sublattice=None):
    """
    Draw a graph obtained from a triangular lattice

    Args:
        graph: a networkx graph whose edges are between nearest neighbor 
            vertices of a triangular lattice
        sublattice: None, 'honeycomb', or 'kagome'. Graph nodes not in the 
            sublattice will appear in red
    """

    if sublattice is None:
        is_favored = lambda x: True
    elif sublattice == 'honeycomb':
        is_favored = lambda x: (x[0] + x[1]) % 3 != 0
    elif sublattice == 'kagome':
        is_favored = lambda x: (x[0] % 2 != 0 or x[1] % 2 != 0)
    else:
        raise ValueError(f'sublattice {sublattice} is not implemented')

    positions = {node: np.array([[1, -1/2], [0, 3**0.5/2]]) @ np.array(node) for node in graph.nodes()}

    max_x = max(value[0] for value in positions.values())
    min_x = min(value[0] for value in positions.values())
    x_size = (max_x - min_x)*0.75
    max_y = max(value[1] for value in positions.values())
    min_y = min(value[1] for value in positions.values())
    y_size = (max_y - min_y)*0.75

    fig = plt.figure(figsize=(x_size, y_size))
    colors = ['red' if not is_favored(node) else 'blue' for node in graph.nodes()]
    nx.draw(graph, positions, node_color=colors)
    plt.show()