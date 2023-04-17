import numpy as np
import pickle
from graph_kernel import triangular_random_walk, dist_from_graph


def prepare_graph_list(N, num_nodes, p=0.0):
    """
    Args:
        N: Total number of graphs
        num_nodes: number of nodes in each graph
        p: probability of accepting a non-preferred vertex to the random walk

    Returns: a list of graphs and an array specifying the class each graph 
        belongs to (either 1 for 'honeycomb' or 0 for 'kagome').
    """
    graph_list = []
    y = np.zeros(N)
    for i in range(N//2):
        graph = triangular_random_walk(num_nodes, p, sublattice='honeycomb')
        graph_list.append(graph)
        y[i] = 1
    for i in range(N//2, N):
        graph = triangular_random_walk(num_nodes, p, sublattice='kagome')
        graph_list.append(graph)
        y[i] = 0
    return graph_list, y

def prepare_distributions(graph_list, y, num_layers, shots=1024):
    N = len(graph_list)
    num_nodes = graph_list[0].number_of_nodes()

    X_all = np.zeros((2**(4*num_layers), N, num_nodes + 1))
    for j, ind in enumerate(np.ndindex((2, 2, 2, 2)*num_layers)):
        discrete_parameters = np.array([list(ind)]).reshape((num_layers, 4))
        X = np.zeros((N, num_nodes + 1))
        for i, graph in enumerate(graph_list):
            X[i] = dist_from_graph(graph, discrete_parameters, shots=shots)
            print(f'Prepared distribution from graph {i + 1}/{N} for parameters '
                f'{j + 1}/{2**(4*num_layers)}' + ' '*30, end='\r')
        X_all[j] = X
    print('\n')

    with open('X_' + f'{num_layers}' + '_layer.pkl', 'wb') as f:
        pickle.dump(X_all, f)

    with open('y.pkl', 'wb') as f:
        pickle.dump(y, f)

    return X_all, y


if __name__ == '__main__':
    graph_list, y = prepare_graph_list(200, 20, p=0.0)

    print('Preparing distributions from 1-layer circuits')
    prepare_distributions(graph_list, y, 1)

    print('Preparing distributions from 2-layer circuits')
    prepare_distributions(graph_list, y, 2)
    
    print('Preparing distributions from 2-layer circuits')
    prepare_distributions(graph_list, y, 3)
