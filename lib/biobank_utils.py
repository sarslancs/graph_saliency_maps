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



# Indexing starts from 1, required for visualising maps
good_components = {'100': [2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
                           18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                           34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 48, 49, 50, 52,
                           53, 57, 58, 60, 63, 64, 93],
        
                    '25': [1,  2,  3,  5,  6,  7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,  
                           18, 19, 20, 21, 22]
                    }
    
# Gender codes
MALE    = 1
FEMALE  = 0

# Default seed if not given by
SEED = 101

# Path to default CSV file unless none provided
CSV_PATH = '/vol/biobank/12579/brain/ukb8972_extracted.csv'

# Data directory if not given
DATA_DIR = '/vol/biobank/12579/brain/fmri'

import os, random, pickle
from functools import partial

import csv
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.spatial import distance
from sklearn.preprocessing import minmax_scale

from scipy.io import loadmat
from scipy.ndimage import zoom
from scipy.ndimage.morphology import binary_fill_holes

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


def generate_train_val_test_IDs(subject_ids, labels, test_ratio=0.2, seed=SEED):
    '''
    Randomly pick IDs for train, test and validation subjects
    '''
    
       
    np.random.seed(seed)
    random.seed(seed)
    
    val_ratio = 0.1
    train_ratio = 1.0 - test_ratio - val_ratio
    
    
    IDs = range(len(labels))
    F_IDs = [IDs[i] for i, label in enumerate(labels) if label == FEMALE]
    M_IDs = [IDs[i] for i, label in enumerate(labels) if label == MALE]
    
    Fs = np.reshape(random.sample(F_IDs, len(F_IDs)), len(F_IDs))
    Ms = np.reshape(random.sample(M_IDs, len(M_IDs)), len(M_IDs))
    
    # Partition datasets
    F_train = int(len(F_IDs) * train_ratio)
    F_test = int(len(F_IDs) * test_ratio)
    
    M_train = int(len(M_IDs) * train_ratio)
    M_test = int(len(M_IDs) * test_ratio)
    
    # Arrange train IDs in a way that two sequential IDs have different genders
    train_IDs = np.full((F_train + M_train), -1)
    F_IDs = Fs[0:F_train]
    M_IDs = Ms[0:M_train]
    for i in range(M_train):
        train_IDs[i*2] = M_IDs[i]
    train_IDs[train_IDs == -1] = F_IDs
    

    # No need to arrange these the same as train IDs
    test_IDs = np.concatenate((Fs[F_train:F_train+F_test], 
                               Ms[M_train:M_train+M_test]))
    val_IDs = np.concatenate((Fs[F_train+F_test:], Ms[M_train+M_test:]))
        
    return train_IDs, test_IDs, val_IDs


def compute_adjacency_and_save(conn_tag='25751_2_0', metric='correlation', 
                               indexing='column', k=10, label_name='31-0.0',
                               remove_ids=None, save=False):
    '''
    Compute adjacency for all subjects available and save
    '''
    
    # conn_tag look-up
    #25750-2.0	rfMRI full correlation matrix, dimension 25
    #25751-2.0	rfMRI full correlation matrix, dimension 100
    #25752-2.0	rfMRI partial correlation matrix, dimension 25
    #25753-2.0	rfMRI partial correlation matrix, dimension 100
    #25754-2.0	rfMRI component amplitudes, dimension 25
    #25755-2.0	rfMRI component amplitudes, dimension 100
    
    dims = {'25750_2_0': 25, '25751_2_0': 100, 
            '25752_2_0': 25, '25753_2_0': 100,
            '25754_2_0': 25, '25755_2_0': 100}
    
    subject_ids = get_ids(conn_tag=conn_tag) 
    _, subject_ids = get_subject_label(subject_ids, label_name=label_name)
    vectors = load_all_vectors(subject_ids, tag=conn_tag)
    
    N = compute_graph_structure_for_networks(vectors, metric, indexing, k=k,
                                             remove_ids=remove_ids)
    
    if save:
        save_to = '/vol/medic02/users/sa1013/CNN_GRAPH/cnn_graph/codebase/data/biobank_gender/'        
        save_name = save_to + 'extended_ica_' + str(dims[conn_tag]) + '_' + \
                    conn_tag + '_' + indexing + '_k' + str(k)
        if remove_ids is not None:
            id_tag = ''.join(['_{}'.format(i) for i in remove_ids])
            save_name = save_name + id_tag
        with open(save_name + '_N.pkl', 'w') as f:
            pickle.dump(N, f)
    
    return N
    

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


def get_all_templates(d=100):
    '''get_all_templates
    '''
    parc_name = 'rfMRI_ICA_d' + str(d) + '.nii.gz'    
    template = root + 'data/biobank_gender/UKBiobank_BrainImaging_GroupMeanTemplates/T1.nii.gz'
    
    # Indexing starts from 1
    valid_ids = [a-1 for a in good_components[str(d)]]

    t1_nii = nib.load(template)
    ica_nii = nib.load(root + data_dir + parc_name)
    ica_valid = ica_nii.get_data()[:,:,:,valid_ids]
    return ica_valid, ica_nii, t1_nii

def save_ICA_maps(ica_nii):
    '''save_ICA_maps
    '''
    valid_ids = [a-1 for a in good_components[str(ica_nii.shape[-1])]]
    ica = ica_nii.get_data()[:,:,:,valid_ids]
    # Save ICA maps        
    for i in range(ica.shape[-1]):
        img = nib.Nifti1Image(ica[:,:,:,i], ica_nii.affine)
        nib.save(img, root + 'tmp/ica_{}.nii.gz'.format(i+1))
    
def map_data_to_brain(data, name, d=100, res=2):
    '''map_data_to_brain
    '''
    
    # Load templates
    ica, ica_nii, nii = get_all_templates(d)
    if res == 1:    
        accum_maps = np.zeros(ica.shape[:3])    
    else:
        accum_maps = np.zeros(nii.get_data().shape)        
    for i in range(ica.shape[-1]):
        ica_map = ica[:,:,:,i]
        ica_map[ica_map < 0] = 0 # mask out negative values
        rescaled_map = ica_map / np.max(ica_map) # scale remaining vals
        rescaled_map *= data[i] # Populate with data[i] based on ica weights         
        if res > 1: # Upsample
            rescaled_map = zoom(rescaled_map, 2, order=1) # Reshape to original space           
        accum_maps += rescaled_map
        # Save individual maps
        # img = nib.Nifti1Image(rescaled_map, nii.affine)
        # nib.save(img, root + 'vis/' + name + '_comp_{}_{}mm.nii.gz'.format(i+1, res))
    
    img = nib.Nifti1Image(accum_maps, nii.affine)
    nib.save(img, root + 'vis/' + name + '_map_{}mm.nii.gz'.format(res))
    img = nib.Nifti1Image(accum_maps / np.max(accum_maps), nii.affine)
    nib.save(img, root + 'vis/' + name + '_scaled_map_{}mm.nii.gz'.format(res))
    
def map_temporal_data_to_brain(data, name, d=100, res=2):
    '''map_data_to_brain
    '''
    
    # Load templates
    ica, ica_nii, nii = get_all_templates(d)
     
    if res == 1:      
        longi = np.zeros((ica.shape[0], ica.shape[1], 
                          ica.shape[2], data.shape[0]))
    else:
        longi = np.zeros((nii.get_data().shape[0], nii.get_data().shape[1], 
                          nii.get_data().shape[2], data.shape[0]))
    for t in range(data.shape[0]):
        if res == 1:    
            accum_maps = np.zeros(ica.shape[:3])    
        else:
            accum_maps = np.zeros(nii.get_data().shape)    
        for i in range(ica.shape[-1]):
            ica_map = ica[:,:,:,i]
            ica_map[ica_map < 0] = 0 # mask out negative values
            rescaled_map = ica_map / np.max(ica_map) # scale remaining vals
            rescaled_map *= data[t, i] # Populate with data[i] based on ica weights         
            if res > 1: # Upsample
                rescaled_map = zoom(rescaled_map, 2, order=1) # Reshape to original space           
            accum_maps += rescaled_map
        
        longi[:,:,:,t] = accum_maps
        
    img = nib.Nifti1Image(longi, nii.affine)
    nib.save(img, root + 'vis/' + name + '_{}mm.nii.gz'.format(res))




