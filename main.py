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

from lib import config_utils

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
    
    
    # Read and print the config file   
    conf_dict = config_utils.read_config_file(args.config)
    config_utils.print_config(conf_dict)