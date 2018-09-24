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

import pickle

from graph_utils import (perm_data, grid_graph, coarsen_adjacency, 
                         compute_laplacians)

from data_utils import map_permed_data, compute_cam_heatmap

from sklearn import preprocessing

import matplotlib.pylab as plt

MNIST_PIXELS = 28

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



def generate_graph_structure(conf_dict, graph_file=None):
    '''
    Read parameters from conf_dict and generate graph 
    structure and apply coarsening. If load_from given, load from disk
    
    Parameters
    ----------
    conf_dict : dict
        dictionary holding all parameters
    load_from : str
        path to graph struct
        
    Returns
    -------
    graph_struct : dict 
        dictionary holding laplacians plus other matrices that might come 
        handy at some point (including, adjacency, graphs, perm)
    '''
    
    if graph_file != None:
        print('graph_struct exists, loading from disk...')
        with open(graph_file) as f:  # Python 3: open(..., 'rb')
            graph_struct = pickle.load(f)      
    else:        
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
        
        print('Graph and Laplacian matrix have been computed. Saving to disk... ')   
        with open(conf_dict['data_dir'] + '/graph.pkl' , 'w') as f:
            pickle.dump(graph_struct, f)
        
    return graph_struct


def cam_multiple_images(X_test, y_test, label, n, graph_struct, model):
    '''
    Compute cam for multiple MNIST images
    '''
    
    all_labels = np.unique(y_test, return_counts=True)[1]
    
    err = 'n cannot be greater than {} (# of images with label {})'.format(all_labels[label], label)
    assert all_labels[label] > n, err
    
    ids = np.where(y_test == label)[0][0:n]
    labels = y_test[ids]
    data = X_test[ids,:]
    
    cam_all = np.zeros((784, n))
    
    predictions, cam_convs, cam_weights = model.get_cam_multiple(data, labels)

    for i, id_ in enumerate(ids[:n]):       
        # Compute cams
        cam_conv, p = cam_convs[i,], predictions[i]
        cam, cam_conv_maps = compute_cam_heatmap(cam_conv, cam_weights, 
                                                 p, graph_struct)
    
        cam_norm = preprocessing.MinMaxScaler().fit_transform(np.expand_dims(cam,1))  
        cam_all[:,i] = cam_norm.flatten()
           
    return cam_all, np.mean(cam_all, axis=1)


def show_cam_heatmap(instance, cam, interpolation='bilinear'):
    cam_im = np.reshape(cam, [MNIST_PIXELS, MNIST_PIXELS])
    real_im = np.reshape(instance, [MNIST_PIXELS, MNIST_PIXELS])

    plt.figure()
    f, axarr = plt.subplots(1, 3, figsize=(12,4))
    
    # Original
    axarr[0].imshow(real_im, cmap='gray')
    axarr[0].set_title('Image')
    
    # CAM
    axarr[1].imshow(cam_im, cmap=plt.cm.jet, alpha=1.0, 
                     interpolation=interpolation)
    axarr[1].set_title('CAM')
    
    # Overlay
    axarr[2].imshow(real_im, cmap='gray', interpolation='none')
    axarr[2].imshow(cam_im, cmap=plt.cm.jet, interpolation=interpolation, 
                     alpha=0.5)
    axarr[2].set_title('Overlay')
    
    
    for i in range(3):
        axarr[i].set_xticks([]) 
        axarr[i].set_yticks([]) 