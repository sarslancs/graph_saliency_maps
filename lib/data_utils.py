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

from biobank_utils import (get_ids, get_subject_labels, load_all_vectors,
                           square_all_vectors, generate_train_val_test_IDs)

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
        
def map_permed_data(perm_nodes, graph_struct):
    '''
    Removes fake nodes (hence, fake grads) added by model and restores 
    original data
    
    '''

    perm = graph_struct['perm']
    adjacency = graph_struct['adjacency']
    
    # Reshape data for compatibility
    data = np.reshape(perm_nodes, len(perm))
    
    M = adjacency.shape[0]
    data_map = np.zeros(adjacency.shape[0])
    for i, j in enumerate(perm):
        if j < M:
            data_map[j] = data[i]
            
    return data_map


def map_permed_data_multiple(perm_nodes, graph_struct):
    '''
    Call map_permed_data n times
    
    '''

    perm = graph_struct['perm']
    adjacency = graph_struct['adjacency']
    
    if perm_nodes.shape[0] == len(perm):
        perm_nodes = perm_nodes.transpose()
    
    n = perm_nodes.shape[0]
    data_mapped = np.zeros((n, adjacency.shape[0]))
    
    for i in range(n):
        data_mapped[i] = map_permed_data(perm_nodes[i], graph_struct)
            
    return data_mapped

def map_permed_data_nd(perm_nodes, graph_struct):
    '''
    Removes fake nodes (hence, fake grads) added by model and restores 
    original data where d > 1
    
    '''

    perm = graph_struct['perm']
    adjacency = graph_struct['adjacency']
    
    M = adjacency.shape[0]
    data_map = np.zeros((adjacency.shape[0], perm_nodes.shape[1]))
    for i, j in enumerate(perm):
        if j < M:
            data_map[j,:] = perm_nodes[i,:]
            
    return data_map


def apply_op_to_grads(grads, op='pos', scaleafter=False):
    '''
    Apply a neg, max or abs operation to grads. Also handle instance for d > 1 
    '''
    
    if op == 'pos':
        grads = np.max(grads, axis=1) / grads.max()
    elif op == 'neg':
        grads = np.max(-grads, axis=1) / grads.min()
    elif op == 'abs':
        grads = np.abs(grads).max(axis=1)
    else: # just return dim
        grads = grads[:,op]
    
    if scaleafter:
        return minmax_scale(grads)
    
    return grads

    
def reshape_one_dimensional_vector(data):
    '''
    Reshape data matrix from (M,) to (M,1)
    '''
    if len(data.shape) < 2:
        return np.reshape(data, (data.shape[0], 1))
    else:
        return data
        

def expand_dimensionality(data, factor=int(1)):
    '''
    Expands dimensionality of dataset by factor. That is, if factor is 1 
    (default), then copy data columns 1 times to the end of data matrix, hence
    expand dim of data from d to 2d. 
    
    Parameters
    ----------
    data : ndarray
        data matrix of size N x d
    factor : int
        expansion factor
        
    Returns
    -------
    data : ndarray
        epanded data matrix of size N x (d + factor x d)
    '''
    
    
    if len(data.shape) == 2:
        data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    
    d = data.shape[2]
      
    data_copy = data.copy()
    np.concatenate((data, data_copy), axis=2)
    for i in range(factor):
        data = np.concatenate((data, data_copy), axis=2)

    assert data.shape[2] == (d + factor * d), 'Data matrix dimensionality mismatch'
    return data
    
def get_data_instance_for_test(data, labels, idx):
    '''
    Returns data instance and label from a dataset for a given idx. Applies
    necessary processing so that returning variables are of size 
    1 x data.shape[1] for data and just 1 for label
    '''
    
    # https://stackoverflow.com/questions/14745199/how-to-merge-two-tuples-in-python/14745275#14745275
    new_data_size = tuple(j for i in (1, data.shape[1:]) for j in (i if isinstance(i, tuple) else (i,)))
    
    assert (idx < data.shape[0]) 
    instance = np.reshape(data[idx,:], new_data_size)

    return instance, labels[idx]

def get_batch_data_by_copying(instance, label, batch_size):
    '''
    Copy values of data instance (of shape 1 x M) into a new ndarray of size 
    batch_size x M. This is a workaround to run predict for just one 
    subject/image.
    '''
    assert(instance.shape[0] == 1)
    
    new_data_size = tuple(j for i in (batch_size, instance.shape[1:]) for j in (i if isinstance(i, tuple) else (i,)))
    
    batch_data = np.zeros(new_data_size)
    batch_label = np.zeros(batch_size)
    
    batch_data[0,:] = instance
    batch_label[0] = label
    for i in range(1, batch_size):
        batch_data[i,:] = instance
        batch_label[i] = label
    
    
    return batch_data , batch_label.astype(np.int32)   

    
def get_biobank_data(conf_dict):
    '''
    Get biobank data all in one go
    '''
 
    # Get variables from config
    conn_tag = conf_dict['conn_tag']
    data_dir = conf_dict['data_dir']
    csv_path = conf_dict['csv_path']
    data_field = conf_dict['data_field']
    num_subjects = conf_dict['num_subjects']
    indexing = conf_dict['indexing'] 
    test_ratio = conf_dict['test_ratio']
    seed = conf_dict['seed']
    
    # Get subject IDs
    subject_ids = get_ids(conn_tag=conn_tag, data_dir=data_dir)[:num_subjects]
    
    # Load labels and update subject_ids based on data field validity
    label_dict, subject_ids = get_subject_labels(subject_ids, data_field, 
                                                 csv_path)
    y_data = np.array([int(label_dict[x]) for x in subject_ids])
    
    # Load data as vectors
    vectors = load_all_vectors(subject_ids, conn_tag, data_dir)
     
    # Convert vectors to matrices, pay attention to index mapping
    X_data = square_all_vectors(vectors, indexing=indexing)
    
    # Split train/test 0.8/0.2 by default
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                        test_size=test_ratio, 
                                                        random_state=seed)
                
    # Split validation/test keep half of test for validation
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, 
                                                    test_size=0.5, 
                                                    random_state=seed)
        
                
    return X_train, y_train, X_val, y_val, X_test, y_test






