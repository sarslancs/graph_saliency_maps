# Copyright (c) 2018 Salim Arslan <salim.arslan@imperial.ac.uk>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import numpy as np

from graph_utils import (perm_data, grid_graph, coarsen_adjacency, 
                         compute_laplacians)

def read_mnist_data(conf_dict, one_hot=False):
    '''
    Read mnist data
    '''
    from tensorflow.examples.tutorials.mnist import input_data
    
    save_to = conf_dict['data_dir']
    return input_data.read_data_sets(save_to, one_hot)


def get_mnist_train_data(mnist, graph_struct=None):
    '''
    Load mnist train data in GCN compatible format (only when graph_struct 
    is provided)
    '''
    
    # Apply perm operation to train_data later during training
    train_data = mnist.train.images.astype(np.float32)
    train_labels = mnist.train.labels.astype(np.int32)
    
    if graph_struct is not None:
        # To make everyting compatible after adding "fake nodes"
        perm = graph_struct['perm']
        adjacency = graph_struct['adjacency']  
        if not adjacency.shape[0] == len(perm):
            train_data = perm_data(train_data, perm)
    
    return train_data, train_labels 


def get_mnist_test_data(mnist, graph_struct=None):
    '''
    Load mnist test data in GCN compatible format (only when graph_struct 
    is provided)
    '''

    test_data = mnist.test.images.astype(np.float32)
    test_labels = mnist.test.labels
    
    if graph_struct is not None:
        # To make everyting compatible after adding "fake nodes"
        perm = graph_struct['perm']
        adjacency = graph_struct['adjacency']
        
        if not adjacency.shape[0] == len(perm):
            test_data = perm_data(test_data, perm)
 
    return test_data, test_labels


def get_mnist_validation_data(mnist, graph_struct=None):
    '''
    Load mnist validation data in GCN compatible format (only when graph_struct 
    is provided)
    '''

    validation_data = mnist.validation.images.astype(np.float32)
    validation_labels = mnist.validation.labels
    
    if graph_struct is not None:
        # To make everyting compatible after adding "fake nodes"
        perm = graph_struct['perm']
        adjacency = graph_struct['adjacency']
        
        if not adjacency.shape[0] == len(perm):
            validation_data = perm_data(validation_data, perm)
 
    return validation_data, validation_labels



def generate_graph_structure(conf_dict):
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
        handy at some point (including, adjacency, graphs, perm)
    '''
    
    
    # Graph parameters
    coarsening_levels   = conf_dict['coarsening_levels']
    number_edges        = conf_dict['number_edges'] 
    metric              = conf_dict['metric'] 
    
    adjacency = grid_graph(m=28, number_edges=number_edges, metric=metric, 
                           corners=False)

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
    graph_struct['adjacency'] = adjacency
    
    print('Graph structure and Laplacian matrix have been computed. ')   
    
    return graph_struct