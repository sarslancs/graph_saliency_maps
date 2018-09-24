# Copyright (c) 2018 Salim Arslan <salim.arslan@imperial.ac.uk>
# Copyright (c) 2016 Michael Defferrard <https://github.com/mdeff/cnn_graph/>
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


import lib.graph_utils as graph
from lib.data_utils import get_batch_data_by_copying
                            
import tensorflow as tf
import sklearn
import scipy.sparse
import numpy as np
import os, time, collections, shutil

from tensorflow.python.framework import ops

class base_model(object):
    
    def __init__(self):
        self.regularizers = []
    
    # High-level interface which runs the constructed computational graph.    
    def get_logits(self, instance, neuron_selector, sess=None):
        sess = self._get_session(sess) # Restore session
        assert (instance.shape[0] == 1)
        batch_data, _ = get_batch_data_by_copying(instance, neuron_selector, 
                                                  self.batch_size)
        feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1, 
                     self.neuron_selector: neuron_selector}
        
        logits = sess.run(self.op_logits, feed_dict=feed_dict)
        max_logits = sess.run(self.op_maxlogit, feed_dict=feed_dict)
        y_logit = sess.run(self.op_y_logit, feed_dict=feed_dict)
        return logits[0], max_logits[0], y_logit
      
    
    def cam_activation(self, data, labels=None, sess=None, verbose=False):
        size = data.shape[0]
        sess = self._get_session(sess)
        predictions = np.empty(size)
        cam_convs_all = None
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            
            if len(data.shape) == 2:
                batch_data = np.zeros((self.batch_size, data.shape[1]))
            else:
                batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
                
            tmp_data = data[begin:end,:]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}
            
            # Compute loss if labels are given.
            batch_labels = np.zeros(self.batch_size)
            batch_labels[:end-begin] = labels[begin:end]
            feed_dict[self.ph_labels] = batch_labels
            batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
            cam_convs, cam_weights = sess.run([self.cam_conv, 
                                               self.cam_fc_value], feed_dict)
            
            if cam_convs_all is None:
               cam_convs_all = np.empty((size, cam_convs.shape[1], 
                                               cam_convs.shape[2])) 
               
            predictions[begin:end] = batch_pred[:end-begin].flatten() # add this for regression to work
            cam_convs_all[begin:end,] = cam_convs[:end-begin]
        
        predictions = predictions.astype(np.int32)

        return predictions, cam_convs_all, cam_weights
        
    def predict(self, data, labels=None, sess=None, verbose=False):
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            
            if len(data.shape) == 2:
                batch_data = np.zeros((self.batch_size, data.shape[1]))
            else:
                batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
                
            tmp_data = data[begin:end,:]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}
            
            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros(self.batch_size)
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)
            
            predictions[begin:end] = batch_pred[:end-begin].flatten() #added this for regression to work
        
        predictions = predictions.astype(np.int32)
  
        if labels is not None:
            if verbose:
                acc = (100.0 * sum(predictions == labels)) / len(labels)
                print('{} out of {} images (%{}) are correctly classified.'
                      .format(sum(predictions == labels), len(labels), acc))

            return predictions, loss * self.batch_size / size
        else:
            return predictions
        
    def predict_once(self, instance, label, sess=None):
        '''
        A workaround for running code for just one subject/image
        '''
        
        assert (instance.shape[0] == 1)
        data, labels = get_batch_data_by_copying(instance, label, 
                                                 self.batch_size)
        predictions, loss = self.predict(data, labels)     
        
        print('label: {}, prediction: {}, loss: {:3f}'.format(label, 
                                                                predictions[0],
                                                                loss))
        return predictions[0], loss
        
     
    def get_cam(self, instance, label, sess=None):
        '''
        A workaround for running code for just one subject/image
        '''
        
        assert (instance.shape[0] == 1)
        data, labels = get_batch_data_by_copying(instance, label, 
                                                 self.batch_size)
        preds, cam_convs, cam_weights = self.cam_activation(data, labels)     

        return preds[0,], cam_convs[0,], cam_weights
    
    
    def get_cam_multiple(self, data, labels, sess=None):
        '''
        Run code for multiple images
        '''
        
        preds, cam_convs, cam_weights = self.cam_activation(data, labels)     

        return preds, cam_convs, cam_weights
    
    def evaluate(self, data, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
        t_wall = time.time()
        predictions, loss = self.predict(data, labels, sess)
        ncorrects = sum(predictions == labels)
        accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
        string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
                accuracy, ncorrects, len(labels), f1, loss)
 
        if sess is None:
            string += '\ntime: {:.0f}s '.format(time.time()-t_wall)
        return string, accuracy, f1, loss
    
    
    def fit(self, train_data, train_labels, val_data, val_labels):
        """
        Runs training and validates it periodically.

        """
        t_wall = time.time()
        sess = tf.Session(graph=self.graph)
        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        train_writer = tf.summary.FileWriter(self._get_path('summaries/train'), self.graph)
        val_writer = tf.summary.FileWriter(self._get_path('summaries/validation'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        try:
            os.makedirs(self._get_path('checkpoints'))
        except OSError, e:
            if e.errno != os.errno.EEXIST:
                raise             
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)

        # Generate subset of train data for eval
        print('Generating a subset of train dataset for monitoring training.')
        start_time = time.time()
        idx_eval = np.random.choice(train_data.shape[0], val_data.shape[0])
        train_data_eval = train_data[idx_eval,:]
        train_labels_eval = train_labels[idx_eval]
        print(' finished in {:.3f} seconds'.format(time.time()-start_time))    
                
        # Training.        
        val_accuracies = []
        val_losses = []
        train_accuracies = []
        train_losses = []
        indices = collections.deque()
        lr_queue = collections.deque(2*[-1], 2)
                
        num_steps = self.num_steps
        for step in range(1, num_steps+1):
            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]

            batch_data, batch_labels = train_data[idx,:], train_labels[idx]
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices

            feed_dict = {self.ph_data: batch_data, 
                         self.ph_labels: batch_labels, 
                         self.ph_dropout: self.dropout, 
                         self.ph_learning_rate: self.learning_rate}
            learning_rate, loss = sess.run([self.op_train, self.op_loss], feed_dict)
                        
            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                print('step {} / {}: lr = {}'.format(step, self.num_steps, 
                                                      learning_rate))
                
                val_string, val_accuracy, val_f1, val_loss = self.evaluate(val_data, val_labels, sess)
                val_accuracies.append(val_accuracy)
                val_losses.append(val_loss)

                train_string, train_acc, train_f1, train_loss = self.evaluate(train_data_eval, train_labels_eval, sess)
                train_accuracies.append(train_acc)
                train_losses.append(train_loss)

                print('  validation {}'.format(val_string))
                print('  training {}'.format(train_string))
                print('  time: {:.0f}s'.format(time.time()-t_wall))
                
                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='accuracy', simple_value=val_accuracy)
                summary.value.add(tag='f1', simple_value=val_f1)
                summary.value.add(tag='loss', simple_value=val_loss)
                val_writer.add_summary(summary, step)
                
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='accuracy', simple_value=train_acc)
                summary.value.add(tag='f1', simple_value=train_f1)
                summary.value.add(tag='loss', simple_value=train_loss)
                train_writer.add_summary(summary, step)
                
                
                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)
                
                # Reduce learning rate if necessary (see paper for details)
                if lr_queue[0] > -1:
                    if lr_queue[1] < lr_queue[0]:
                        if val_accuracy < lr_queue[1]:
                            self.learning_rate *= self.decay_rate

                lr_queue.append(val_accuracy)   
                

        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(val_accuracies), np.mean(val_accuracies)))
        print('training accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(train_accuracies), np.mean(train_accuracies)))
        train_writer.close()
        val_writer.close()
        sess.close()
                
        t_step = (time.time() - t_wall) / num_steps
        return train_accuracies, train_losses, val_accuracies, val_losses, t_step
    
       
    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    # Methods to construct the computational graph.
    def build_graph(self, M_0, d):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Inputs.
            with tf.name_scope('inputs'):
                if d == 1:
                    self.ph_data = tf.placeholder(tf.float32, (self.batch_size, M_0), 'data')
                elif d > 1:
                    self.ph_data = tf.placeholder(tf.float32, (self.batch_size, M_0, d), 'data')

                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')
                self.neuron_selector = tf.placeholder(tf.int32, (), 'neuron_selector')

            # Model.
            self.op_logits = self.inference(self.ph_data, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(self.op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                    self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(self.op_logits)
            self.op_maxlogit = self.maxlogit(self.op_logits)
            self.op_y_logit = self.y_logit(self.op_logits, self.neuron_selector)
            self.op_gradients = self.gradients(self.op_y_logit, self.ph_data)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            
            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=10)
        
        self.graph.finalize()
    
    def inference(self, data, dropout):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout.
            True: the model is built for training.
            False: the model is built for evaluation.
        """

        logits = self._inference(data, dropout)
        return logits
    
    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=1)
            return prediction
        
    def maxlogit(self, logits):
        '''
        Return max value in logits
        '''
        with tf.name_scope('maxlogit'):
            max_logit = tf.reduce_max(logits, reduction_indices=[1])
            return max_logit
        
    def y_logit(self, logits, neuron_selector):
        '''
        Return value indexed with neuron_selector in logits
        '''
        with tf.name_scope('y_logit'):
            return logits[0][neuron_selector]


    def gradients(self, y_logit, data):
        '''
        Return gradients node
        '''
        with tf.name_scope('gradients'):
            gradients_node = tf.gradients(y_logit, data)
            return gradients_node 
    
    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.to_int64(labels)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                model_loss = tf.reduce_mean(cross_entropy)
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = model_loss + regularization
            
            # Summaries for TensorBoard.
            tf.summary.scalar('loss/model_loss', model_loss)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([model_loss, regularization, loss])
                tf.summary.scalar('loss/avg/model_loss', averages.average(model_loss))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average
    
    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, 
                 momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            lr = ops.convert_to_tensor(learning_rate)
            self.ph_learning_rate = tf.placeholder_with_default(lr, (), 'learning_rate')
            tf.summary.scalar('learning_rate', self.ph_learning_rate)
            # Optimizer.
            if momentum == 0:
                optimizer = tf.train.AdamOptimizer(self.ph_learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(self.ph_learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(self.ph_learning_rate, name='control')
            return op_train

    # Helper methods.
    def _get_path(self, folder):
        return os.path.join(self.dir_name, folder)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def bspline_basis(K, x, degree=3):
    """
    Return the B-spline basis.

    K: number of control points.
    x: evaluation points
       or number of evenly distributed evaluation points.
    degree: degree of the spline. Cubic spline by default.
    """
    if np.isscalar(x):
        x = np.linspace(0, 1, x)

    # Evenly distributed knot vectors.
    kv1 = x.min() * np.ones(degree)
    kv2 = np.linspace(x.min(), x.max(), K-degree+1)
    kv3 = x.max() * np.ones(degree)
    kv = np.concatenate((kv1, kv2, kv3))

    # Cox - DeBoor recursive function to compute one spline over x.
    def cox_deboor(k, d):
        # Test for end conditions, the rectangular degree zero spline.
        if (d == 0):
            return ((x - kv[k] >= 0) & (x - kv[k + 1] < 0)).astype(int)

        denom1 = kv[k + d] - kv[k]
        term1 = 0
        if denom1 > 0:
            term1 = ((x - kv[k]) / denom1) * cox_deboor(k, d - 1)

        denom2 = kv[k + d + 1] - kv[k + 1]
        term2 = 0
        if denom2 > 0:
            term2 = ((-(x - kv[k + d + 1]) / denom2) * cox_deboor(k + 1, d - 1))

        return term1 + term2

    # Compute basis for each point
    basis = np.column_stack([cox_deboor(k, degree) for k in range(K)])
    basis[-1,-1] = 1
    return basis


class cgcnn(base_model):
    """
    Graph CNN which uses the Chebyshev approximation.

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of filters.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.
        C: how many conv layers before applying max pooling?
    
    gap: defines if there is a gap between gconv layers and softmax layer       
    
    L: List of Graph Laplacians. Size M x M. One per coarsening level.
    d: length of 'signal' i.e. 3rd dimensionality of tensor (added by SA)
    
    
    The following are choices of implementation for various blocks.
        filter: filtering operation, e.g. chebyshev5, lanczos2 etc.
        brelu: bias and relu, e.g. b1relu or b2relu.
        pool: pooling, e.g. mpool1.
    
    Training parameters:
        num_steps:    Number of training steps.
        learning_rate: Initial learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """
    def __init__(self, L, F, K, p, num_classes, C, d=1, filter='chebyshev5', 
                 brelu='b1relu', pool='mpool1', num_steps=1000, 
                 learning_rate=0.1, decay_rate=0.95, decay_steps=None, 
                 momentum=0.9, regularization=0, dropout=0, batch_size=100, 
                 eval_frequency=200, dir_name=''):
        super(cgcnn, self).__init__()
        
        # Verify the consistency w.r.t. the number of layers.
        assert len(L) >= len(F) == len(K) == len(p)
        assert np.all(np.array(p) >= 1)
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.
        assert len(L) >= 1 + np.sum(p_log2)  # Enough coarsening levels for pool sizes.
        
        # Keep the useful Laplacians only. May be zero.
        M_0 = L[0].shape[0]
        j = 0
        self.L = []
        for pp in p:
            self.L.append(L[j])
            j += int(np.log2(pp)) if pp > 1 else 0
        L = self.L
        
        # Print information about NN architecture.
        Ngconv = len(p)
        print('NN architecture')
        print('  input: M_0 = {}'.format(M_0))
        for i in range(Ngconv):
            print('  layer {0}: cgconv{0} x {1}'.format(i+1, C[i]))
            print('    representation: M_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
                    i, i+1, L[i].shape[0], F[i], p[i], L[i].shape[0]*F[i]//p[i]))
            F_last = F[i-1] if i > 0 else 1
            print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                    i, i+1, F_last, F[i], K[i], F_last*F[i]*K[i]))
            if brelu == 'b1relu':
                print('    biases: F_{} = {}'.format(i+1, F[i]))
            elif brelu == 'b2relu':
                print('    biases: M_{0} * F_{0} = {1} * {2} = {3}'.format(
                        i+1, L[i].shape[0], F[i], L[i].shape[0]*F[i]))

        print(' layer {}: {}'.format(Ngconv+i, 'gap'))
        M_gap = F[-1]
        print('    representation: M_{} = {}'.format(Ngconv+i, 
              M_gap))
        
        print('  layer {}: {}'.format(Ngconv+i+1, 'logits (softmax)'))
        print('    representation: M_{} = {}'.format(Ngconv+i+1, 
              num_classes))
        print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                Ngconv+i, Ngconv+i+1, M_gap, num_classes, num_classes*M_gap))
        print('    biases: M_{} = {}'.format(Ngconv+i+1, num_classes))

        
        # Store attributes and bind operations.
        self.L, self.F, self.K, self.p, self.C = L, F, K, p, C
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)

        
        # Build the computational graph.
        self.build_graph(M_0, d)
        
    def filter_in_fourier(self, x, L, Fout, K, U, W):
        # TODO: N x F x M would avoid the permutations
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        # Transform to Fourier domain
        x = tf.reshape(x, [M, Fin*N])  # M x Fin*N
        x = tf.matmul(U, x)  # M x Fin*N
        x = tf.reshape(x, [M, Fin, N])  # M x Fin x N
        # Filter
        x = tf.matmul(W, x)  # for each feature
        x = tf.transpose(x)  # N x Fout x M
        x = tf.reshape(x, [N*Fout, M])  # N*Fout x M
        # Transform back to graph domain
        x = tf.matmul(x, U)  # N*Fout x M
        x = tf.reshape(x, [N, Fout, M])  # N x Fout x M
        return tf.transpose(x, perm=[0, 2, 1])  # N x M x Fout

    def fourier(self, x, L, Fout, K):
        assert K == L.shape[0]  # artificial but useful to compute number of parameters
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Fourier basis
        _, U = graph.fourier(L)
        U = tf.constant(U.T, dtype=tf.float32)
        # Weights
        W = self._weight_variable([M, Fout, Fin], regularization=False)
        return self.filter_in_fourier(x, L, Fout, K, U, W)

    def spline(self, x, L, Fout, K):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Fourier basis
        lamb, U = graph.fourier(L)
        U = tf.constant(U.T, dtype=tf.float32)  # M x M
        # Spline basis
        B = bspline_basis(K, lamb, degree=3)  # M x K
        #B = bspline_basis(K, len(lamb), degree=3)  # M x K
        B = tf.constant(B, dtype=tf.float32)
        # Weights
        W = self._weight_variable([K, Fout*Fin], regularization=False)
        W = tf.matmul(B, W)  # M x Fout*Fin
        W = tf.reshape(W, [M, Fout, Fin])
        return self.filter_in_fourier(x, L, Fout, K, U, W)

    def chebyshev5(self, x, L, Fout, K):
        
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def b2relu(self, x):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def mpool1(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            #tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def apool1(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.avg_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x
        
    def _inference(self, x, dropout):
        # Graph convolutional layers.
        if len(x.get_shape().as_list()) == 2:    
            x = tf.expand_dims(x, 2)  # N x M x F=1
        
        for i in range(len(self.p)):
            for c in range(self.C[i]):
                with tf.variable_scope('conv{}_{}'.format(i+1,c+1)):
                    with tf.name_scope('filter'):
                        x = self.filter(x, self.L[i], self.F[i], self.K[i])
                    with tf.name_scope('bias_relu'):
                        x = self.brelu(x)   
            with tf.name_scope('dropout'):
                x = tf.nn.dropout(x, dropout)   

            if self.p[i] > 1:    
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.p[i])
            
        self.cam_conv = x # Feature maps from final conv layer 
                
        with tf.variable_scope('cam'): # GAP
            self.gap = tf.reduce_mean(x, axis=[1])
                
            # Logits linear layer, i.e. softmax without normalization.
            self.logits = self.fc(self.gap, self.num_classes, relu=False)
            
        with tf.variable_scope('cam', reuse=True): # CAM
            self.cam_fc_value = tf.nn.bias_add(tf.get_variable('weights'), 
                                               tf.get_variable('bias'))

        return self.logits
