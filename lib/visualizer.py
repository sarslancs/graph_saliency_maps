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


from matplotlib import pyplot as plt 


def monitor_training(t_accuracy, t_loss, v_accuracy, v_loss):
    plt.figure()
    fig, axarr = plt.subplots(2, 1, figsize=(8, 8))
    axarr[0].plot(t_accuracy, 'b.-')
    axarr[0].plot(v_accuracy, 'r.-')
    axarr[0].legend(['train accuracy', 'val accuracy'])        
    axarr[1].plot(t_loss, 'b.-')
    axarr[1].plot(v_loss, 'r.-')
    axarr[1].legend(['train loss', 'val loss'])  
    plt.show()
    
    
    
    