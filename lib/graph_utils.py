import os, pickle
import numpy as np
import scipy
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial import distance
from matplotlib import pyplot as plt

"""Utility functions for graph CNNs.
Adapted from https://github.com/mdeff/cnn_graph/tree/master/lib
"""

def alter_graph_structure(adj=None, ids=[4, 20], eps=0.001, level=0):
    '''alter_graphs
    '''
    
    assert(level <= 2)
    
    if isinstance(adj, scipy.sparse.csr.csr_matrix):
        adj_full = adj.todense()
    else:
        adj_full = adj
    for i in ids:
        to_alter = np.where(adj_full[i,:]>0)[-1]
        for node in to_alter:
            adj[i, node] = eps
            adj[node, i] = eps
            
    
    if level == 1:
        for i in ids:
            to_alter_0 = np.where(adj_full[i,:]>0)[-1]
            for node_0 in to_alter_0:
                to_alter_1 = np.where(adj_full[node_0,:]>0)[-1]
                for node_1 in to_alter_1:
                    adj[node_0, node_1] = eps
                    adj[node_1, node_0] = eps
                    
                    
    if level == 2:
        for i in ids:
            to_alter_0 = np.where(adj_full[i,:]>0)[-1]
            for node_0 in to_alter_0:
                to_alter_1 = np.where(adj_full[node_0,:]>0)[-1]
                for node_1 in to_alter_1:
                    adj[node_0, node_1] = eps
                    adj[node_1, node_0] = eps
                    to_alter_2 = np.where(adj_full[node_1,:]>0)[-1]
                    for node_2 in to_alter_2:
                        adj[node_1, node_2] = eps
                        adj[node_2, node_1] = eps
        
    
    return adj

# TODO: Ideally this should be in a Graph class
def generate_graph_structure(conf_dict, verbose=False):
    '''
    Read parameters from conf_dict and generate graph 
    structure and apply coarsening
    
    Parameters
    ----------
    conf_dict : dict
        dictionary holding all parameters
        
    Returns
    -------
    graph_struct : dict 
        dictionary holding laplacians plus other matrices that might come 
        handy at some point (including, adjacency, graphs, perm
    '''
    
    
    # Get graph-related variables
    graph_name = conf_dict['graph_path']
    
    if graph_name is not None:
        print('graph_struct exists, loading from disk...')
        with open(graph_name) as f:  # Python 3: open(..., 'rb')
            graph_struct = pickle.load(f)        
    else:
        # Graph parameters
        adjacency_path      = conf_dict['adjacency_path']
        coarsening_levels   = conf_dict['coarsening_levels']
        number_edges        = conf_dict['number_edges'] # only needed if adj_path is None
        metric              = conf_dict['metric'] # only needed if adj_path is None
        
        if adjacency_path is None:
#            if 'networks' in graph_struct.keys(): # we are doing UK biobank
#                networks = conf_dict['networks']
#                adjacency = compute_graph_structure_for_networks(networks,
#                                                                 metric)
#            else:
            adjacency = grid_graph(m=28, number_edges=number_edges,
                                       metric=metric, corners=False)
        else:
            adjacency = load_adjacency(adjacency_path)
        
        #adjacency[adjacency>0] = 1
#        saved = adjacency[4,20]
#        ids = [4]
#        ids = [1, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 18, 20, 21, 24, 25, 26, 
#               27, 28, 30, 32, 33, 34, 35, 36, 39, 40, 41, 42, 43, 45, 46, 47, 51, 54] -> test acc down 73
#        adjacency = alter_graph_structure(adjacency, ids, eps=0, level=0)
#        adjacency[4,20] = saved
#        adjacency[20,4] = saved

        
#        ids = [4, 6, 8, 10, 12, 16, 27, 28, 32, 42, 43, 44, 48, 49, 53]
#        for i in ids:
#            adjacency[20,i] = 0.001
#            adjacency[i,20] = 0.001
##            neighs = np.where(adj[i,:] > 0)[-1]
##            for neigh in neighs:
##                adjacency[neigh,i] = 0.001
##                adjacency[i,20] = 0.001
#        #4    
#        ids = [6, 7, 8, 11, 12, 16, 19, 20, 29, 35, 38, 39, 45, 48, 52, 54]
#        for i in ids:
#            adjacency[4,i] = 0.001
#            adjacency[i,4] = 0.001
#            
#        #6
#        ids = [ 4,  5,  8,  9, 10, 12, 20, 29, 31, 35, 42, 44, 48, 52, 54]
#        for i in ids:
#            adjacency[6,i] = 0.001
#            adjacency[i,6] = 0.001
#            
#        #52
#        ids = [ 0,  4,  6,  9, 10, 12, 14, 18, 35, 46, 48, 54]
#        for i in ids:
#            adjacency[52,i] = 0.001
#            adjacency[i,52] = 0.001
            
        graph_struct = {}
        if coarsening_levels > 0:        
            graphs, perms, parents = coarsen_adjacency(adjacency, 
                                                       coarsening_levels)
            graph_struct['graphs'] = graphs
            graph_struct['perm'] = perms[0]
            graph_struct['perms'] = perms
            graph_struct['parents'] = parents
        
        else:
            graphs = []
            for i in range(len(conf_dict['conv_depth'])):
                graphs.append(adjacency)
        
        laplacians = compute_laplacians(graphs)
        
        graph_struct['laplacians'] = laplacians
        graph_struct['adjacency'] = adjacency.copy()
        
        name_supervertices = conf_dict['name_supervertices']
        n_supervertices = conf_dict['n_supervertices']
        
        #Save data
        print('graph_struct is computed. Saving to disk...')
        save_path = conf_dict['data_dir'] + '/'     
        if name_supervertices[:3] == 'ica':
            with open(save_path + '/gt_struct_' + \
                      conf_dict['conn_tag'] + '_' + \
                      name_supervertices + '_' + \
                      str(n_supervertices) + '_coarseby_' + \
                      str(coarsening_levels) + '.pkl', 'w') as f:
                pickle.dump(graph_struct, f)
        elif name_supervertices == 'ho':
            with open(save_path + '/gt_struct_' + \
                      name_supervertices + '_' + \
                      conf_dict['method'] + '_' + \
                      conf_dict['conn_tag'] + '_' + \
                      str(n_supervertices) + '_coarseby_' + \
                      str(coarsening_levels) + '.pkl', 'w') as f:
                pickle.dump(graph_struct, f)
        else:    
            with open(save_path + '/gt_struct_' + \
                      name_supervertices + '_' + \
                      conf_dict['conn_tag'] + '_' + \
                      str(n_supervertices) + '_coarseby_' + \
                      str(coarsening_levels) + '.pkl', 'w') as f:
                pickle.dump(graph_struct, f)
     
    if verbose:
        plot_adjacency(graph_struct['adjacency'])   
        plot_spectrum(graph_struct['laplacians'])
        plt.show()
    return graph_struct


    
def load_adjacency(adj_name):
    '''
    Load adjacency graph from a pickle file (should be laready provided)
    '''
    
    if os.path.exists(adj_name):   
        if adj_name.split('.')[1] == 'mat':
            from scipy.io import loadmat
            data = loadmat(adj_name)
            N = data['N']
        elif adj_name.split('.')[1] == 'pkl':
            try:
                with open(adj_name) as f:  # Python 3: open(..., 'rb')
                    N = pickle.load(f)
            except OSError, e:
                if e.errno != os.errno.EEXIST:
                    raise  
        
        if not isinstance(N, scipy.sparse.csr.csr_matrix):
            return scipy.sparse.csr_matrix(N.astype(np.float32))
        return N.astype(np.float32)

def coarsen_adjacency(adjacency, coarsening_levels=4):
    ''' 
    Coarsen adjacency graph and return coarsened graphs and 
    permutations (perms) that allow traversing them
    '''
    
    graphs, perms, parents = coarsen(adjacency, levels=coarsening_levels)
    return graphs, perms, parents


def compute_laplacians(graphs):
    ''' 
    Compute laplacians for each coarsening level
    '''
    
    laplacians = [laplacian(A, normalized=True) for A in graphs]
    return laplacians

def plot_adjacency(adjacency):
    '''
    Plot plot_adjacency
    '''

    print('d = |V| = {}, k|V| < |E| = {}'.format(adjacency.shape[0], np.sum(adjacency>0)))
    plt.spy(adjacency, markersize=2, color='black')
    

def plot_spectrum(L, algo='eig'):
    """
    Plot the spectrum of a list of multi-scale Laplacians L.
    """
    # Algo is eig to be sure to get all eigenvalues.
    plt.figure(figsize=(12, 5))
    for i, lap in enumerate(L):
        lamb, U = fourier(lap, algo)
        step = 2**i
        x = range(step//2, L[0].shape[0], step)
        lb = 'L_{} spectrum in [{:1.2e}, {:1.2e}]'.format(i, lamb[0], lamb[-1])
        plt.plot(x, lamb, '.', label=lb)
    plt.legend(loc='best')
    plt.xlim(0, L[0].shape[0])
    plt.ylim(ymin=0)
    
    
def fourier(L, algo='eigh', k=1):
    """
    Return the Fourier basis, i.e. the EVD of the Laplacian.
    """

    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]

    if algo is 'eig':
        lamb, U = np.linalg.eig(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo is 'eigh':
        lamb, U = np.linalg.eigh(L.toarray())
    elif algo is 'eigs':
        lamb, U = scipy.sparse.linalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo is 'eigsh':
        lamb, U = scipy.sparse.linalg.eigsh(L, k=k, which='SM')

    return lamb, U
    
def grid_graph(m=28, number_edges=8, metric='euclidean', corners=False):
    ''' 
    Create the regular image grid that is going to define the graph structure 
    '''
    z = grid(m)
    dist, idx = distance_sklearn_metrics(z, k=number_edges, metric=metric)
    A = adjacency(dist, idx)

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        import scipy.sparse
        A = A.toarray()
        A[A < A.max()/1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    print("|E| = {} > k|V| = {}".format(A.nnz//2, number_edges*m**2//2))
    return A


def distance_sklearn_metrics(z, k=4, metric='euclidean'):
    """
    Compute exact pairwise distances.
    """
    
    d = pairwise_distances(z, metric=metric, n_jobs=-2)
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]
    return d, idx


def grid(m, dtype=np.float32):
    """Return the embedding of a grid graph."""
    M = m**2
    x = np.linspace(0, 1, m, dtype=dtype)
    y = np.linspace(0, 1, m, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), dtype)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z




def adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0

    # Weights.
    sigma2 = np.mean(dist[:, -1])**2
    dist = np.exp(- dist**2 / sigma2)

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W


def replace_random_edges(A, noise_level):
    """Replace randomly chosen edges by random edges."""
    M, M = A.shape
    n = int(noise_level * A.nnz // 2)

    indices = np.random.permutation(A.nnz//2)[:n]
    rows = np.random.randint(0, M, n)
    cols = np.random.randint(0, M, n)
    vals = np.random.uniform(0, 1, n)
    assert len(indices) == len(rows) == len(cols) == len(vals)

    A_coo = scipy.sparse.triu(A, format='coo')
    assert A_coo.nnz == A.nnz // 2
    assert A_coo.nnz >= n
    A = A.tolil()

    for idx, row, col, val in zip(indices, rows, cols, vals):
        old_row = A_coo.row[idx]
        old_col = A_coo.col[idx]

        A[old_row, old_col] = 0
        A[old_col, old_row] = 0
        A[row, col] = 1
        A[col, row] = 1

    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    assert type(L) is scipy.sparse.csr.csr_matrix
    return L


def coarsen(A, levels, self_connections=False):
    """
    Coarsen a graph, represented by its adjacency matrix A, at multiple
    levels.
    """
    graphs, parents = metis(A, levels)
    perms = compute_perm(parents)

    for i, A in enumerate(graphs):
        M, M = A.shape

        if not self_connections:
            A = A.tocoo()
            A.setdiag(0)

        if i < levels:
            A = perm_adjacency(A, perms[i])

        A = A.tocsr()
        A.eliminate_zeros()
        graphs[i] = A

        Mnew, Mnew = A.shape
        print('Layer {0}: M_{0} = |V| = {1} nodes ({2} added),'
              '|E| = {3} edges'.format(i, Mnew, Mnew-M, A.nnz//2))

    return graphs, perms, parents
#    return graphs, perms[0] if levels > 0 else None


def metis(W, levels, rid=None):
    """
    Coarsen a graph multiple times using the METIS algorithm.

    INPUT
    W: symmetric sparse weight (adjacency) matrix
    levels: the number of coarsened graphs

    OUTPUT
    graph[0]: original graph of size N_1
    graph[2]: coarser graph of size N_2 < N_1
    graph[levels]: coarsest graph of Size N_levels < ... < N_2 < N_1
    parents[i] is a vector of size N_i with entries ranging from 1 to N_{i+1}
        which indicate the parents in the coarser graph[i+1]
    nd_sz{i} is a vector of size N_i that contains the size of the supernode in the graph{i}

    NOTE
    if "graph" is a list of length k, then "parents" will be a list of length k-1
    """

    N, N = W.shape
    if rid is None:
        rid = np.random.permutation(range(N))
    parents = []
    degree = W.sum(axis=0) - W.diagonal()
    graphs = []
    graphs.append(W)

    for _ in range(levels):
        # CHOOSE THE WEIGHTS FOR THE PAIRING
        weights = degree            # graclus weights
        weights = np.array(weights).squeeze()

        # PAIR THE VERTICES AND CONSTRUCT THE ROOT VECTOR
        idx_row, idx_col, val = scipy.sparse.find(W)
        perm = np.argsort(idx_row)
        rr = idx_row[perm]
        cc = idx_col[perm]
        vv = val[perm]
        cluster_id = metis_one_level(rr,cc,vv,rid,weights)  # rr is ordered
        parents.append(cluster_id)

        # COMPUTE THE EDGES WEIGHTS FOR THE NEW GRAPH
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1
        # CSR is more appropriate: row,val pairs appear multiple times
        W = scipy.sparse.csr_matrix((nvv,(nrr,ncc)), shape=(Nnew,Nnew))
        W.eliminate_zeros()
        # Add new graph to the list of all coarsened graphs
        graphs.append(W)
        N, N = W.shape

        # COMPUTE THE DEGREE (OMIT OR NOT SELF LOOPS)
        degree = W.sum(axis=0)

        # CHOOSE THE ORDER IN WHICH VERTICES WILL BE VISTED AT THE NEXT PASS
        ss = np.array(W.sum(axis=0)).squeeze()
        rid = np.argsort(ss)

    return graphs, parents


# Coarsen a graph given by rr,cc,vv.  rr is assumed to be ordered
def metis_one_level(rr,cc,vv,rid,weights):

    nnz = rr.shape[0]
    N = rr[nnz-1] + 1

    marked = np.zeros(N, np.bool)
    rowstart = np.zeros(N, np.int32)
    rowlength = np.zeros(N, np.int32)
    cluster_id = np.zeros(N, np.int32)

    oldval = rr[0]
    count = 0
    clustercount = 0

    for ii in range(nnz):
        rowlength[count] = rowlength[count] + 1
        if rr[ii] > oldval:
            oldval = rr[ii]
            rowstart[count+1] = ii
            count = count + 1

    for ii in range(N):
        tid = rid[ii]
        if not marked[tid]:
            wmax = 0.0
            rs = rowstart[tid]
            marked[tid] = True
            bestneighbor = -1
            for jj in range(rowlength[tid]):
                nid = cc[rs+jj]
                if marked[nid]:
                    tval = 0.0
                else:
                    tval = vv[rs+jj] * (1.0/weights[tid] + 1.0/weights[nid])
                if tval > wmax:
                    wmax = tval
                    bestneighbor = nid

            cluster_id[tid] = clustercount

            if bestneighbor > -1:
                cluster_id[bestneighbor] = clustercount
                marked[bestneighbor] = True

            clustercount += 1

    return cluster_id


def compute_perm(parents):
    """
    Return a list of indices to reorder the adjacency and data matrices so
    that the union of two neighbors from layer to layer forms a binary tree.
    """

    # Order of last layer is random (chosen by the clustering algorithm).
    indices = []
    if len(parents) > 0:
        M_last = max(parents[-1]) + 1
        indices.append(list(range(M_last)))

    for parent in parents[::-1]:
        # Fake nodes go after real ones.
        pool_singeltons = len(parent)

        indices_layer = []
        for i in indices[-1]:
            indices_node = list(np.where(parent == i)[0])
            assert 0 <= len(indices_node) <= 2

            # Add a node to go with a singelton.
            if len(indices_node) is 1:
                indices_node.append(pool_singeltons)
                pool_singeltons += 1
            # Add two nodes as children of a singelton in the parent.
            elif len(indices_node) is 0:
                indices_node.append(pool_singeltons+0)
                indices_node.append(pool_singeltons+1)
                pool_singeltons += 2

            indices_layer.extend(indices_node)
        indices.append(indices_layer)

    # Sanity checks.
    for i,indices_layer in enumerate(indices):
        M = M_last*2**i
        # Reduction by 2 at each layer (binary tree).
        assert len(indices[0] == M)
        # The new ordering does not omit an indice.
        assert sorted(indices_layer) == list(range(M))

    return indices[::-1]

def perm_data_nd(x, indices):
    """
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    Do this for multi dimnesional data matrices
    """
    if indices is None:
        return x

    N, M, D = x.shape
    Mnew = len(indices)
    assert Mnew >= M, 'There should always be at least M nodes!'
    xnew = np.empty((N, Mnew, D))
    for i,j in enumerate(indices):
        # Existing vertex, i.e. real data.
        if j < M:
            xnew[:,i,:] = x[:,j,:]
        # Fake vertex because of singletons.
        # They will stay 0 so that max pooling chooses the singleton.
        else:
            xnew[:,i,:] = np.zeros((N,D))
    return xnew


def perm_data(x, indices):
    """
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return x
    
    # If there exists a 3rd dim call perm_data_nd 
    if len(x.shape) > 2:
        return perm_data_nd(x, indices)
        
    N, M = x.shape
    Mnew = len(indices)
    assert Mnew >= M
    xnew = np.empty((N, Mnew))
    for i,j in enumerate(indices):
        # Existing vertex, i.e. real data.
        if j < M:
            xnew[:,i] = x[:,j]
        # Fake vertex because of singletons.
        # They will stay 0 so that max pooling chooses the singleton.
        else:
            xnew[:,i] = np.zeros(N)
    return xnew


def perm_adjacency(A, indices):
    """
    Permute adjacency matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return A

    M, M = A.shape
    Mnew = len(indices)
    assert Mnew >= M
    A = A.tocoo()

    # Add Mnew - M isolated vertices.
    if Mnew > M:
        rows = scipy.sparse.coo_matrix((Mnew-M,    M), dtype=np.float32)
        cols = scipy.sparse.coo_matrix((Mnew, Mnew-M), dtype=np.float32)
        A = scipy.sparse.vstack([A, rows])
        A = scipy.sparse.hstack([A, cols])

    # Permute the rows and the columns.
    perm = np.argsort(indices)
    A.row = np.array(perm)[A.row]
    A.col = np.array(perm)[A.col]

    # assert np.abs(A - A.T).mean() < 1e-9
    assert type(A) is scipy.sparse.coo.coo_matrix
    return A

def distance_scipy_spatial(z, k=4, metric='euclidean'):
    """Compute exact pairwise distances."""
    d = scipy.spatial.distance.pdist(z, metric)
    d = scipy.spatial.distance.squareform(d)
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]
    
    return d, idx


