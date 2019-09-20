# SimultCodClass
Pytorch code for simultaneous classifier learning and sparse coding
This package includes the code used for the following experiments in the paper "Supervised learning with incomplete data via sparse representations", submitted to ICLR 2020:
 - Synthetic dataset: here, we applied our simultaneous method, with hyperparameters $\lambda_1$ and $\lambda_2$ tuned via cross-validation, to train a logistic regression classifier on the training dataset with randomly distributed missing entries, and compared the obtained Test Accuracy against the following standard sequential methods: 
{\bf Seq. Sparse}: reconstructions are obtained by finding the sparsest representation compatible with the observations solving a LASSO problem;  
{\bf Zero Fill}: missing entries are filled with zeros, which is equivalent to ignore unknown values;
{\bf Mean Unsupervised}: missing entries are filled with the mean computed on the available values in the same position in the rest of data samples;
{\bf Mean Supervised}: as in the previous case but the mean is computed on the samples of the same class vectors only;
{\bf K-Nearest Neighbor (KNN)}: as in the previous case but the mean is computed on the K-Nearest Neighbors of the same class only
 - MNIST dataset with a CNN4 classifier
 
 Dependencies:
 - Pytorch 1.0.0
 
# SimultCodClass
