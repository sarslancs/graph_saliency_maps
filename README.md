
# Graph Saliency Maps through Spectral Convolutional Networks

The code in this repository provides the implementation of an activation-based visual attribution method for irregular graphs, which works integrated with graph convolutional neural networks (GCNs). The method has been validated via a sex classification task using functional brain connectivity networks and data from the [UK Biobank](http://www.ukbiobank.ac.uk/) and is presented in our paper: 

- **Salim Arslan**, Sofia Ira Ktena, Ben Glocker, Daniel Rueckert, [Graph Saliency Maps through Spectral Convolutional Networks: Application to Sex Classification with Brain Connectivity](https://arxiv.org/abs/1806.01764)

The paper is presented at [Second International Workshop on Graphs in Biomedical Image Analysis](https://grail-miccai.github.io/), ornaised as a part of [Medical Image Computing and Computer-Assisted Interventions (MICCAI), 2018, Granada](https://miccai2018.org/en/Default.asp?). 

Here are the [slides](#), [poster](https://www.researchgate.net/profile/Salim_Arslan/publication/327751019_Poster/data/5ba24ac945851574f7d66901/arslan-salim-poster-A0.pdf) (which has won the best poster award!) and a copy of the [paper](https://arxiv.org/abs/1806.01764). The visual abstract is provided below: 

![Graph saliency maps with spectral convolutional networks](http://gdurl.com/HACf)

The code is released under the terms of the [MIT license](https://github.com/sarslancs/graph_saliency_maps/blob/master/licence.txt). Please cite the above paper if you use it.

Our implementation is integrated to the spectral convolutional network codebase provided in [this repository](https://github.com/mdeff)

## Results
Click the below picture or scan the QR code to see the method on action (you will be directed to our Youtube channel).
[![Audi R8](http://gdurl.com/yHO9G)](https://www.youtube.com/watch?v=F7K-8P-OcRs "Graph saliency maps with spectral convolutional networks")

## Installation
1.  Clone this repository.
	```
    git clone https://github.com/sarslancs/graph_saliency_maps.git
    cd graph_saliency_maps
	```
   
2.  Install the dependencies. Please edit  `requirements.txt`  to choose the [TensorFlow](https://www.tensorflow.org/install/) version (CPU / GPU, Linux / Mac) you need to install, or install it before attempting to use the code. 

	```
	pip install -r requirements.txt  # or make install
	```

3. The codebase is based on `Python 2.7.14, Anaconda custom (64-bit)`. We tested the code on `tensorflow 1.3.0` , `tensorflow-gpu 1.4.1`, and `tensorflow-tensorboard 0.1.8` on a workstation running `Ubuntu 16.04`. The CUDA version was `release 8.0, V8.0.61`. At the time of release, versions of other libraries were as follows: `numpy 1.12.1, sklearn 0.19.1, scipy 1.1.0, matplotlib 2.2.2`.

## How to?
In order to get started, you need to provide:

 1. A config file, where all model hyper-parameters, data paths, and training/test parameters are specified. While we have provided one, it depends on the UK Biobank data, which although is freely available, may not be directly obtained online. You have to first register your intent to use the data and undergo an application process. 
 2. Datasets for training, validation, and test of dimensionality `num_subjects x num_regions x num_signals`.  
 3. An adjacency matrix of size `num_regions x num_regions` that represents the underlying graph and is subsequently used to compute the Laplacian matrix. If not provided, the code computes one based on the train data.

To run the model in training mode:
 - `python main.py -c ./config/gender_biobank.conf`

To run the model in test mode (i.e. via a pre-trained model):
 - `python main.py -c ./config/gender_biobank.conf -m
   ./log/sex_biobank_model_2018-09-24-13-01`

## Contact
https://twitter.com/salimarslan

	
	
