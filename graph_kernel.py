import networkx as nx
import numpy as np
from stabilizer_tools import StabilizerTableau
import time

def dist_from_graph(graph, num_layers=3, shots=100):

    qubit_mapping = {}
    for i, node in enumerate(graph.nodes):
        qubit_mapping[node] = i

    num_qubits = graph.number_of_nodes()



    time1 = time.time()
    
    time2 = time.time()

    
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

        # for node in graph.nodes():
        #     q = qubit_mapping[node]
        #     S.conjugate('h', q)
        #     S.conjugate('s', q)
        # for edge in graph.edges():
        #     source = qubit_mapping[edge[0]]
        #     target = qubit_mapping[edge[1]]
        #     S.conjugate('cx', source, target)
        #     S.conjugate('s', target)
        #     S.conjugate('cx', source, target)
        #     S.conjugate('cz', source, target)
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

    time3 = time.time()

    results = S.sample_z_basis(destabilizers=D, shots=shots)

    time4 = time.time()

    occupation_numbers = np.zeros(num_qubits + 1)
    for key, value in results.items():
        occupation_numbers[key.count('1')] += value

    time5 = time.time()

    # print(f'time initializing: {time2 - time1}')
    # print(f'time conjugating: {time3 - time2}')
    # print(f'time sampling: {time4 - time3}')
    # print(f'time getting distribution: {time5 - time4}')

    return occupation_numbers/shots

def shannon(dist):
    return -np.sum(dist*np.log(dist, where=(dist!=0)))

def jensen_shannon(dist1, dist2):
    return shannon((dist1 + dist2)/2) - (shannon(dist1) + shannon(dist2))/2

def kernel_entry(graph1, graph2, coeff=40, shots=100):
    dist1 = dist_from_graph(graph1, shots=shots)
    dist2 = dist_from_graph(graph2, shots=shots)
    length = max(len(dist1), len(dist2))
    dist1 = np.pad(dist1, ((0, length - len(dist1))))
    dist2 = np.pad(dist2, ((0, length - len(dist2))))
    return np.exp(-coeff*jensen_shannon(dist1, dist2))

def kernel(X1, X2, coeff=40, shots=100):
    kernel = np.zeros((len(X1), len(X2)))
    for i, j in np.ndindex(kernel.shape):
        kernel[i, j] = kernel_entry(X1[i], X2[j], coeff=coeff, shots=shots)
    return kernel