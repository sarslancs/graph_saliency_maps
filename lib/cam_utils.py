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

from functools import partial
import numpy as np

from data_utils import get_data_instance_for_test, compute_cam_heatmap

def argmax_k(data_2d, k=4, d=55):
    '''
    argmax operation for k top indices
    '''
    
    if data_2d.shape[0] != d:
        data_2d = data_2d.transpose()
        
    listed = [data_2d[:,i] for i in range(data_2d.shape[1])] 
    
    mapped = map(partial(lambda arr, k : arr.argsort()[-k:][::-1], k=k), 
                 listed)  
    
    items = []
    for arr in mapped: [items.append(item) for item in arr]
    
    return items, mapped


def compute_roi_frequency(argmax_out, d=55):
    '''
    Compute number of times a value appears in argmax_out 
    '''
    freqs = np.zeros(d)
    for m in argmax_out:
        freqs[m] += 1        
    return freqs

def cam_single_image(X_test, y_test, graph_struct, idx, model):
    '''
    Compute class activations for a single images
    '''
    
    instance, label = get_data_instance_for_test(X_test, y_test, idx)
    pred, cam_convs, cam_weights = model.get_cam(instance, label) 
    
    
    # Compute CAM
    cam, cam_conv_maps = compute_cam_heatmap(cam_convs, cam_weights, pred, 
                                             graph_struct)
    
    return cam, cam_conv_maps, cam_weights, pred

def cam_multiple_images(X_test, y_test, label, model, graph_struct=None):                                   
    '''
    Compute class activations for multiple images
    '''

    ids = np.where(y_test == label)[0]
    labels = y_test[ids]
    data = X_test[ids,:]
    
    cam_all = np.zeros((graph_struct['adjacency'].shape[0], len(ids)))
    
    predictions, cam_convs, cam_weights = model.get_cam_multiple(data, labels)

    for i, id_ in enumerate(ids):       
        # Compute cams
        cam_conv, pred = cam_convs[i,], predictions[i]
        cam, cam_conv_maps = compute_cam_heatmap(cam_conv, cam_weights, 
                                                 pred, graph_struct)
        cam_all[:,i] = cam
    
    return cam_all, np.mean(cam_all, axis=1)