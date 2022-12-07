# ECE-6122_CUDA_MLP

Dataset can be downloaded from http://yann.lecun.com/exdb/mnist/

Download into dataset/ folder and unzip the files.

To run the code, first compile with "make cuda" in the main directory. To run the program type the following command with the following arguments in the terminal: "./nn [number of hidden units] [batch\_size]  [number of training epochs (0 for inference only)] [load weights (bool)]". The user can specify the number of hidden units in the network, batch size, number of training epochs, and whether to load weights saved periodically during the training process. The weights are stored in the folder "current\_weights". 
