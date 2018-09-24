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

import argparse

import models

import os

from lib.visualizer import monitor_training
from lib.config_utils import (read_config_file, print_config, 
                              load_params_from_config)
from lib.data_utils import get_biobank_data
from lib.graph_utils import generate_graph_structure

from lib.cam_utils import argmax_k, compute_roi_frequency, cam_multiple_images

def train(X_train, y_train, X_val, y_val, model, verbose=False):
    '''
    Train model and show some stats
    '''
    t_accuracy, t_loss, v_accuracy, v_loss, t_step = model.fit(X_train, 
                                                               y_train, 
                                                               X_val, 
                                                               y_val)
    if verbose:
        print('Close figure to continue in terminal mode...')
        monitor_training(t_accuracy, t_loss, v_accuracy, v_loss)
        
    return t_accuracy, t_loss, v_accuracy, v_loss, t_step 

def test(X_test, y_test, model, verbose=False):
    '''
    Test a model 
    '''
    predictions, loss = model.predict(X_test, y_test, verbose=verbose)
    return predictions, loss


def get_args():
    parser = argparse.ArgumentParser(description='Graph saliency maps with GCNs')
    parser.add_argument('-c', '--config', dest='config', required=True,
                      default='./config/gender_biobank.conf', type=str,
                      help='config file to start processing')
    parser.add_argument('-m', '--model', dest='model_path', required=False,
                      default=None, type=str, help='path to pre-trained model')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    # Get args
    args = get_args()
    
    # Check if a model is given and if yes, it is valid
    if args.model_path != None:
        assert os.path.exists(args.model_path), \
              args.model_path + ' does not exist'
        print('Running in test mode...')
        print('')
    else:
        print('Running in training mode...')
        print('')
       
    # Read and print the config file   
    conf_dict = read_config_file(args.config)
    print_config(conf_dict)
    
    # Load data based on config file. 
    # Change the function according to your data setup.
    # X_Data should of of shape (n, dx, dy), with n, dx, dy being the number 
    # of subjects, nodes, and features, respectively
    (X_train, y_train, 
     X_val, y_val, 
     X_test, y_test) = get_biobank_data(conf_dict)
        
    # Generate underyling graph structure W and Laplacian matrix L
    graph_struct = generate_graph_structure(conf_dict, X_train)
    laplacians = graph_struct['laplacians']
    
    # Load model parameters for GCN     
    params = load_params_from_config(conf_dict, len(X_train), args.model_path)
    
    # Build the GCN model
    model = models.cgcnn(laplacians, **params)
    
    # Train a model if no model provided
    if args.model_path == None:
        print('Training has started...')
        t_accuracy, t_loss, _, _, _ = train(X_train, y_train, X_val, y_val, 
                                            model, True)    
    
    # Test a model
    print('Testing model...')
    predictions, loss = test(X_test, y_test, model, True)
    
    # Acquire class avtivations for all test subjects
    print('Computing CAMs for all subjects...')
    cams_op_0, _ = cam_multiple_images(X_test, y_test, 0, model,
                                       graph_struct=graph_struct)    
    cams_op_1, _ = cam_multiple_images(X_test, y_test, 1, model,
                                       graph_struct=graph_struct)
    
    # Obtain population-level saliency maps 
    print('Generating population-level saliency maps ..')
    counts, _ = argmax_k(cams_op_0, k=3, d=conf_dict['d'])
    graph_saliency_0 = compute_roi_frequency(counts, d=conf_dict['d'])
    counts, _ = argmax_k(cams_op_1, k=3, d=conf_dict['d'])
    graph_saliency_1 = compute_roi_frequency(counts, d=conf_dict['d'])