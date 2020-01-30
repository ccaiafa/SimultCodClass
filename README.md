# SimultCodClass
Pytorch code for simultaneous classifier learning and sparse coding
This package includes the code used for the following experiments in the paper "Simultaneous Training of Neural Networks and Sparse Coding with Incomplete Data", submitted to ICML 2020:
 - **Synthetic dataset:** we compare our simultaneous method with standard imputation methods: **Seq. Sparse**: reconstructions are obtained by finding the sparsest representation compatible with the observations solving a LASSO problem;  **Zero Fill**: missing entries are filled with zeros, which is equivalent to ignore unknown values; **Mean Unsupervised**: missing entries are filled with the mean computed on the available values in the same position in the rest of data samples; **Mean Supervised**: as in the previous case but the mean is computed on the samples of the same class vectors only; **K-Nearest Neighbor (KNN)**: as in the previous case but the mean is computed on the K-Nearest Neighbors of the same class only
 
 - **MNIST dataset with a CNN4 classifier**
 
 - **CIFAR10 dataset with a ResNet18 classifier**
 
 Dependencies:
 - Pytorch 1.0.0
 
# SimultCodClass
