# DeepMEns: An ensemble model for predicting of sgRNA on-target activity based on multiple features
![Figure2](https://github.com/user-attachments/assets/6bfd7db6-b4d4-4037-873c-10d835e890a5)
## Overview
The overall framework of DeepMEns. (a) Partitioning of the training datasets. The training dataset is further partitioned into five parts; each part is used to regulate the hyper-parameters, and the remaining four parts are used for training. (b) There are three neural network frameworks. The first framework processes one-hot encoding combined with 0-1 representation of the secondary structure feature. The second framework processes the DNA shape feature matrix. The third framework processes the positional encoding feature matrices. (c) The independent test datasets are inputted into the five models with different weights and biases to compare the performance by using the average ensemble learning approach.
## Environment
    The algorithm mainly uses the following packages:
    conda create -n DeepMEns python=3.7
    conda activate DeepMEns
    pip install tensorFlow==2.4.1
    pip install keras==2.10.0
    pip install numpy==1.19.5
    pip install pandas==1.3.5
