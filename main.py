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

from lib.config_utils import read_config_file, print_config
from lib.data_utils import get_biobank_data
from lib.graph_utils import generate_graph_structure

def get_args():
    parser = argparse.ArgumentParser(description='Graph saliency maps using GCNs')
    parser.add_argument('-c', '--config', dest='config', required=True,
                      default='./config/gender_biobank.conf', type=str,
                      help='config file to start processing')
        
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    # Get args
    args = get_args()
       
#    # Read and print the config file   
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