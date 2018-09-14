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


def argmax_k(data_2d, k=4, d=55):
    
    if data_2d.shape[0] != d:
        data_2d = data_2d.transpose()
        
    listed = [data_2d[:,i] for i in range(data_2d.shape[1])] 
    
    mapped = map(partial(lambda arr, k : arr.argsort()[-k:][::-1], k=k), 
                 listed)  
    
    items = []
    for arr in mapped: [items.append(item) for item in arr]
    
    return items, mapped

def map_temporal_ica_to_brain(preds, name, d=100, res=2):
    '''map_data_to_brain
    '''
    
    # Load templates
    ica, ica_nii, nii = get_all_templates(100)
    mask = nii.get_data()
    mask[mask<200] = 0
    mask[mask>200] = 1  
    
    _, mapped = argmax_k(preds, k=4)
    
    if res == 1:      
        longi = np.zeros((ica.shape[0], ica.shape[1], 
                          ica.shape[2], len(mapped)))
    else:
        longi = np.zeros((nii.get_data().shape[0], nii.get_data().shape[1], 
                          nii.get_data().shape[2], len(mapped)))
    
    for t, ids in enumerate(mapped):
        if res == 1:    
            accum_maps = np.zeros(ica.shape[:3])    
        else:
            accum_maps = np.zeros(nii.get_data().shape) 
        for i in ids:
            ica_map = ica[:,:,:,i]
            ica_map[ica_map < 0] = 0 # mask out negative values
            rescaled_map = ica_map #/ np.max(ica_map) # scale remaining vals    
            if res > 1: # Upsample
                rescaled_map = zoom(rescaled_map, 2, order=1) # Reshape to original space           
                rescaled_map[mask < 1] = 0
            accum_maps += rescaled_map
        rescaled_map = accum_maps / np.max(accum_maps)    
        longi[:,:,:,t] = rescaled_map
        
    img = nib.Nifti1Image(longi, nii.affine)
    nib.save(img, root + 'vis/' + name + '_{}mm.nii.gz'.format(res))
    
#    img = nib.Nifti1Image(accum_maps / np.max(accum_maps), nii.affine)
#    nib.save(img, root + 'vis/' + name + '_scaled_map_{}mm.nii.gz'.format(res))
    
def load_freqs_back(num_folds=10, num_runs=10, num_parc=55, 
                    model='simple', scaler=None):
    '''load_data_back
    '''
    grads_freqs_all_0 = np.zeros((num_parc, num_folds, num_runs))
    grads_freqs_all_1 = np.zeros((num_parc, num_folds, num_runs))
    cams_freqs_all_0 = np.zeros((num_parc, num_folds, num_runs))
    cams_freqs_all_1 = np.zeros((num_parc, num_folds, num_runs))
    accs_all = np.zeros((num_runs, num_folds))
    
    for i in range(num_runs):
        if model == 'simple':
            load_name = root + 'results/grads_cams_freqs_ops_partial_simple_' + str(num_folds) + '_fold_' + str(i) + '.mat'
        else:
            load_name = root + 'results/grads_cams_freqs_ops_partial_' + str(num_folds) + '_fold_' + str(i) + '.mat'
        mat = loadmat(load_name) 
        accs_all[i,:] =  mat['test_accs']    
        grads_freqs_all_0[:,:,i] = mat['grads_freqs_0'] 
        grads_freqs_all_1[:,:,i] = mat['grads_freqs_1'] 
        cams_freqs_all_0[:,:,i] = mat['cams_freqs_0'] 
        cams_freqs_all_1[:,:,i] = mat['cams_freqs_1'] 
        
        if i == 0:
            grads_op_1 = mat['grads_op_list_0'][0][0][i]
            grads_op_0 = mat['grads_op_list_1'][0][0][i]
            cams_op_0 = mat['cams_op_list_0'][0][0][i]
            cams_op_1 = mat['cams_op_list_1'][0][0][i]
        else:
            grads_op_1 = np.r_[grads_op_1, mat['grads_op_list_0'][0][0][i]]
            grads_op_0 = np.r_[grads_op_0, mat['grads_op_list_1'][0][0][i]]
            cams_op_0 = np.r_[cams_op_0, mat['cams_op_list_0'][0][0][i]]
            cams_op_1 = np.r_[cams_op_1, mat['cams_op_list_1'][0][0][i]]
        
        
    grads_freqs_0 = np.sum(np.sum(grads_freqs_all_0, axis=2), axis=1)
    grads_freqs_1 = np.sum(np.sum(grads_freqs_all_1, axis=2), axis=1)
    cams_freqs_0 = np.sum(np.sum(cams_freqs_all_0, axis=2), axis=1)
    cams_freqs_1 = np.sum(np.sum(cams_freqs_all_1, axis=2), axis=1)
       
    if scaler == 'max':
        grads_freqs_0 /= np.max(grads_freqs_0)
        grads_freqs_1 /= np.max(grads_freqs_1)
        cams_freqs_0 /= np.max(cams_freqs_0)
        cams_freqs_1 /= np.max(cams_freqs_1)
    elif scaler == 'minmax':
        grads_freqs_0 = minmax_scale(grads_freqs_0)
        grads_freqs_1 = minmax_scale(grads_freqs_1)
        cams_freqs_0 = minmax_scale(cams_freqs_0)
        cams_freqs_1 = minmax_scale(cams_freqs_1)
        
    return grads_freqs_0, cams_freqs_0, grads_freqs_1, cams_freqs_1
  
    
def load_ops_back(num_folds=10, num_runs=10, num_parc=55, 
                  model='simple', scaler=None):
    '''load_data_back
    '''

    for i in range(num_runs):
        if model == 'simple':
            load_name = root + 'results/grads_cams_freqs_ops_partial_simple_' + str(num_folds) + '_fold_' + str(i) + '.mat'
        else:
            load_name = root + 'results/grads_cams_freqs_ops_partial_' + str(num_folds) + '_fold_' + str(i) + '.mat'
        mat = loadmat(load_name) 
        
        if i == 0:
            grads_op_1 = mat['grads_op_list_0'][0][0][i]
            grads_op_0 = mat['grads_op_list_1'][0][0][i]
            cams_op_0 = mat['cams_op_list_0'][0][0][i]
            cams_op_1 = mat['cams_op_list_1'][0][0][i]
        else:
            grads_op_1 = np.r_[grads_op_1, mat['grads_op_list_0'][0][0][i]]
            grads_op_0 = np.r_[grads_op_0, mat['grads_op_list_1'][0][0][i]]
            cams_op_0 = np.r_[cams_op_0, mat['cams_op_list_0'][0][0][i]]
            cams_op_1 = np.r_[cams_op_1, mat['cams_op_list_1'][0][0][i]]
        
        
    grads_mean_0 = np.mean(grads_op_0, 0)
    grads_mean_1 = np.mean(grads_op_1, 0) 
    cams_mean_0 = np.mean(cams_op_0, 0)
    cams_mean_1 = np.mean(cams_op_1, 0) 
    
    if scaler == 'max':
        grads_mean_0 /= np.max(grads_mean_0)
        grads_mean_1 /= np.max(grads_mean_1)
        cams_mean_0 /= np.max(cams_mean_0)
        cams_mean_1 /= np.max(cams_mean_1)
    elif scaler == 'minmax':
        grads_mean_0 = minmax_scale(grads_mean_0)
        grads_mean_1 = minmax_scale(grads_mean_1)
        cams_mean_0 = minmax_scale(cams_mean_0)
        cams_mean_1 = minmax_scale(cams_mean_1)
       
    return grads_mean_0, cams_mean_0, grads_mean_1, cams_mean_1


def get_aal(zoom_level=2, order=1, save=True):
    '''get all
    '''
    
    template = root + 'data/biobank_gender/UKBiobank_BrainImaging_GroupMeanTemplates/T1.nii.gz'
    nii = nib.load(template)
    
    aal_name = root + 'data/biobank_gender/aal/aal2.nii.gz'
    aal_nii = nib.load(aal_name)
    
    parc = aal_nii.get_data()
    if zoom_level > 1:
        parc = zoom(parc, zoom=zoom_level, order=1)
        parc = fill_holes_iteratively(parc)
    
    if save:
        img = nib.Nifti1Image(parc, nii.affine)
        save_name = root + 'data/biobank_gender/aal/' + \
                           'aal2_zoom{}.nii.gz'.format(zoom_level)
        nib.save(img, save_name)
    
    return parc, aal_nii


def save_given_regions(labels=[]):
    '''save_given_regions
    '''
    
    aal, aal_nii = get_aal(zoom_level=1, save=False)
    mask_all = np.zeros(aal.shape).astype(aal.dtype)
    for label in labels:
        mask = (aal == label).astype(aal.dtype)
        mask[aal == label] = label
        mask_all[aal == label] = label
        
        img = nib.Nifti1Image(mask, aal_nii.affine)
        save_name = root + 'data/biobank_gender/aal/' + \
                           'aal2_{}.nii.gz'.format(label)
        nib.save(img, save_name)
    
    whole_mask = mask_all.copy()
    whole_mask[aal > 0] = 121
    whole_mask[mask_all > 0] = mask_all[mask_all > 0]
    
    img = nib.Nifti1Image(whole_mask, aal_nii.affine)
    post = ''
    for i in labels: post += '_{}'.format(i)
    save_name = root + 'data/biobank_gender/aal/' + \
                       'aal2{}.nii.gz'.format(post)
    nib.save(img, save_name)
        
    return True


def fill_holes_iteratively(parc):
    '''fill_holes_iteratively
    '''
    labels = np.arange(1, len(np.unique(parc)))
    filled_parc = np.zeros(parc.shape)
    for label in labels:
        mask = (parc == label)
        filled = binary_fill_holes(mask).astype('int32')
        filled_parc[mask] = filled[mask]
        filled_parc[mask] = label
        
    return filled_parc

 
def combine_ICA_maps_into_RSNs(ica_nii):
    '''combine_ICA_maps_into_RSNs
    '''
    rsns = [[5, 7, 49, 9, 13, 21], 
            [11, 12, 55, 19, 36, 53, 15, 45, 47], 
            [10, 40, 43, 15, 54], 
            [4, 42, 41, 34, 39, 18, 37, 52, 24, 25, 26], 
            [1, 3, 16, 8, 14], 
            [2, 22, 23, 27, 38, 6, 35, 30, 20, 32], 
            [17, 31, 46, 50, 28, 44, 48, 29, 33]]
    
    valid_ids = [a-1 for a in good_components[str(ica_nii.shape[-1])]]
    ica = ica_nii.get_data()[:,:,:,valid_ids]
    # Load ICA maps        
    for i, rsn in enumerate(rsns):
        rsn_map = np.zeros((ica.shape[0], ica.shape[1], ica.shape[2])).astype(np.float32)
        for label in rsn:
            label -= 1
            ica_map = ica[:,:,:,label]
            rsn_map += ica_map
            
        img = nib.Nifti1Image(rsn_map, ica_nii.affine)
        nib.save(img, root + 'tmp/rsn_map_{}.nii.gz'.format(i+1))    

if __name__ == '__main__':
    print('Hello')
    
#    subject_ids = sorted(get_ids())
#    label_dict, subject_ids = get_subject_label(subject_ids, label_name='31-0.0')
#    y_data = np.array([int(label_dict[x]) for x in subject_ids])
#    ages = get_subject_age(subject_ids)
    
#    _, ica_nii, _ =  get_all_templates(d=100)
#    combine_ICA_maps_into_RSNs(ica_nii)
    
    N = compute_adjacency_and_save(conn_tag='25753_2_0', indexing='column', 
                                   k=10, label_name='20016-2.0', save=True)
  
#    parc = get_aal(order=3)
    
#    freqs_0, _, freqs_1, _ = load_freqs_back(model=None)
#    ops_0, _, ops_1, _ = load_ops_back(model=None)
#    map_data_to_brain(freqs_0, res=2, name='grads_freqs_female')
#    map_data_to_brain(freqs_1, res=2, name='grads_freqs_male')
#    map_data_to_brain(ops_0, res=2, name='grads_ops_female')
#    map_data_to_brain(ops_1, res=2, name='grads_ops_male')
    
#    freqs_0, _, freqs_1, _ = load_freqs_back(model='simple', scaler='max')
#    ops_0, _, ops_1, _ = load_ops_back(model='simple', scaler='max')
#    map_data_to_brain(freqs_0, res=1, name='grads_freqs_scaled_max_female')
#    map_data_to_brain(freqs_1, res=1, name='grads_freqs_scaled_max_male')
#    map_data_to_brain(ops_0, res=1, name='grads_ops_scaled_max_female')
#    map_data_to_brain(ops_1, res=1, name='grads_ops_scaled_max_male')
#    
#    
#    freqs_0, _, freqs_1, _ = load_freqs_back(model='simple', scaler='minmax')
#    ops_0, _, ops_1, _ = load_ops_back(model='simple', scaler='minmax')
#    map_data_to_brain(freqs_0, res=2, name='grads_freqs_scaled_minmax_female')
#    map_data_to_brain(freqs_1, res=2, name='grads_freqs_scaled_minmax_male')
#    map_data_to_brain(ops_0, res=2, name='grads_ops_scaled_minmax_female')
#    map_data_to_brain(ops_1, res=2, name='grads_ops_scaled_minmax_male')
    
#    map_data_to_brain(cams_0, res=1, name='cams_freqs_partial_female')
#    map_data_to_brain(cams_1, res=1, name='cams_freqs_partial_male')
    
#    save_given_regions(labels = [5,   6,   3,   4,  19,  71,  72,  89,  37,  69,  38,  20])
    
    
#    root_dir = '/vol/medic02/users/sa1013/CNN_GRAPH/cnn_graph/codebase/results/'
#    freq_mat = loadmat(root_dir + 'freq_mat_for_figures.mat')
#    cam_mat = loadmat(root_dir + 'cam_mat_for_figures.mat')
#    
#    for k in range(4):
#        map_data_to_brain(freq_mat['freqs_1'][:,k], res=2, name='cams_freqs_1_k{}'.format(k+1))
    
#    for i in range(3):
#        map_data_to_brain(cam_mat['cam1_{}'.format(i)].flatten(), res=2, name='cam1_{}'.format(i))    
#    map_data_to_brain(freqs_1, res=2, name='grads_freqs_male')
#    map_data_to_brain(ops_0, res=2, name='grads_ops_female')
#    map_data_to_brain(ops_1, res=2, name='grads_ops_male')
    
#    root_dir = '/vol/medic02/users/sa1013/CNN_GRAPH/cnn_graph/codebase/results/'
#    cam_mat = loadmat(root_dir + 'cams_2.mat')
#    map_temporal_data_to_brain(cam_mat['activations'], res=2, name='temporal_cam_2')  
#    map_temporal_ica_to_brain(cam_mat['activations'], res=2, name='temporal_regions_2') 
#    _, mapped = argmax_k(cam_mat['activations'], k=4)