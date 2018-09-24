# Copyright (c) 2018 Salim Arslan <salim.arslan@imperial.ac.uk>
# Copyright (c) 2017 Sofia Ira Ktena 
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

   
# Default seed if not given by
SEED = 101

# Path to default CSV file unless none provided
CSV_PATH = '/vol/biobank/12579/brain/ukb8972_extracted.csv'

# Data directory if not given
DATA_DIR = '/vol/biobank/12579/brain/fmri'

import os

import csv

import numpy as np

from scipy.spatial import distance

from graph_utils import distance_scipy_spatial, adjacency
       
        
def get_ids(num_subjects=None, conn_tag='25751_2_0', data_dir=DATA_DIR):
    """
        num_subjects   : number of subject IDs to get

    return:
        subject_IDs    : list of subject IDs (length num_subjects)
    """
    
    root_folder = os.path.join(data_dir, conn_tag)
    onlyfiles = [f for f in os.listdir(root_folder)
                 if os.path.isfile(os.path.join(root_folder, f))]
    subject_ids = [f.split('_')[0] for f in onlyfiles]

    if num_subjects is not None:
        subject_ids = subject_ids[:num_subjects]

    return subject_ids


def get_subject_labels(subject_list, label_name='31-0.0', csv_path=CSV_PATH):
    """
        subject_list : the subject short IDs list
        label_name   : name of the label to be retrieved, '31-0.0' for gender

    returns:
        label        : dictionary of subject labels
    """

    label = {}
    valid_ids = []
    with open(os.path.join(csv_path)) as csvfile:
        reader_base = csv.DictReader(csvfile)
        for row in reader_base:
            if row['eid'] in subject_list:
                label[row['eid']] = row[label_name]
                valid_ids.append(row['eid'])

    
    return label, valid_ids


def load_all_vectors(subject_list, tag='25751_2_0', data_dir=DATA_DIR):
    """
        subject_list : the subject short IDs list
        tag          : connectivity field tag

    returns:
        all_vectors : list of connectivity vectors 
    """
    
    root_folder = os.path.join(data_dir, tag)
    
    all_vectors = []
    
    for subject in subject_list:
        fl = os.path.join(root_folder, subject + "_" + tag + ".txt")
        matrix = np.loadtxt(fl)

        all_vectors.append(matrix)

    return all_vectors

def square_all_vectors(vectors, indexing='column', d=55, remove_ids=None):
    """
        vectors     : list of connectivity vectors (only upper triangle)
        indexing     : how to do broadcasting? default column

    returns:
        _ : list of connectivity matrices (regions x regions, symmetric)
    """
    networks = []
    if indexing == 'column': # Because indexing is different in Matlab and python
        idx = row_to_columnwise_indexing(d)
        for vector in vectors:
            net = vector_to_square(vector, d, idx) 
            if remove_ids is not None:
                net = np.delete(net, remove_ids, 0)
                net = np.delete(net, remove_ids, 1)               
            networks.append(net)
    else:
        for vector in vectors:
            net = distance.squareform(vector)
            if remove_ids is not None:
                net = np.delete(net, remove_ids, 0)
                net = np.delete(net, remove_ids, 1)               
            networks.append(net)           
    return np.array(networks)


def vector_to_square(vector, d=55, idx=None):
    square_net_from_file = np.zeros((d, d))
    tr = np.triu(np.ones((d, d)), k=1)
    if idx is None:
        square_net_from_file[tr>0] = vector
    else:
        square_net_from_file[idx] = vector
    return square_net_from_file + square_net_from_file.transpose()


def row_to_columnwise_indexing(d=55):
    idx = ([],[])
    for i in range(d-1):
        for j in range(i+1):
            idx[0].append(j)
            idx[1].append(i+1)
            
    return idx


def compute_graph_structure_for_networks(vectors, metric='correlation',
                                         indexing='column', d=55, k=10,
                                         remove_ids=None):
    """
        vectors    : list of connectivity matrices (only upper triangle)

    returns:
        A : adjacency matrix (regions x regions, symmetric)
    """
    
    mean_net = np.mean(np.array(vectors), axis=0)
    if indexing == 'column':
        idx = row_to_columnwise_indexing(d)
        d = vector_to_square(mean_net, d, idx)
    else:
        d = distance.squareform(mean_net)
    
    if not remove_ids is None:
        d = np.delete(d, remove_ids, 0)
        d = np.delete(d, remove_ids, 1)
        
    dist, idx = distance_scipy_spatial(d, k=k, metric=metric)
    A = adjacency(dist, idx).astype(np.float32)

    return A




