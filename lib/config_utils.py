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

import os

def read_config_file(conf_path):
    '''
    Read config file and return a dictionary with all params
    '''
    
    conf_dict = {}; # create config dict
    exec(open(conf_path).read(), conf_dict)
    return conf_dict


def print_config(conf_dict, save=True):
    '''
    Print all variables in config dict
    '''
    
    if save:
        log_dir = conf_dict['log_dir']
   
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
        with open(log_dir + '/parameters', 'w') as file:
            for key in conf_dict.keys():
                if not key in ['__builtins__', '__doc__', 'os']:
                    string = " %-20s => %-15s" % (key, conf_dict[key])
                    print string
                    file.write(string + '\n')
            file.close() 
    else:
        for key in conf_dict.keys():
            if not key in ['__builtins__', '__doc__', 'os']:
                print(" %-20s => %-15s" % (key, conf_dict[key]))



def load_params_from_config(conf_dict, n_train, trained_model_dir=None):
    
    params = dict()
    
    params['num_steps']      = conf_dict['num_steps']
    params['num_epochs']     = conf_dict['num_epochs']
    params['batch_size']     = conf_dict['batch_size']
    params['eval_frequency'] = conf_dict['eval_frequency']
    
    # Building blocks.
    params['filter']         = conf_dict['filt']
    params['brelu']          = conf_dict['bias']
    params['pool']           = conf_dict['pool']
    
    # Architecture.
    params['F']              = conf_dict['filters']  # Number of graph convolutional filters.
    params['K']              = conf_dict['K_order']  # Polynomial orders.
    params['p']              = conf_dict['strides']  # Pooling sizes.
    params['M']              = conf_dict['num_fc']   # Output dimensionality of fully connected layers.
    params['d']              = conf_dict['d']        # Signal length
    params['C']              = conf_dict['conv_depth']   
    
    # Optimization.
    params['regularization'] = conf_dict['regularization'] 
    params['dropout']        = conf_dict['dropout'] 
    params['learning_rate']  = conf_dict['learning_rate'] 
    params['decay_rate']     = conf_dict['decay_rate'] 
    params['momentum']       = conf_dict['momentum'] 
    params['decay_steps']	= n_train / params['batch_size']
    params['seed']           = conf_dict['seed']    
    
    
    if trained_model_dir == None:
        params['dir_name'] = conf_dict['log_dir']
    else:
        params['dir_name'] = trained_model_dir
    
    return params
