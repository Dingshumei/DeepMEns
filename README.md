# DeepMEns: an ensemble model for predicting of sgRNA on-target activity based on multiple features
![Figure2](https://github.com/user-attachments/assets/6bfd7db6-b4d4-4037-873c-10d835e890a5)
## Overview
The overall framework of DeepMEns. a.) Partitioning of the training datasets. The training dataset is further partitioned into five parts where one part is used to regulate the hyper-parameters, and the remaining four parts are used for training. b.) There are three neural network frameworks. The first framework processes one-hot encoding combined with 0-1 representation of the secondary structure feature. The second framework processes the DNA shape feature matrix. The third framework processes the positional encoding feature matrices. c.) The independent test datasets are inputted into the five models with different weights and biases to compare the performance by using the average ensemble learning approach.
## Environment
    The algorithm mainly uses the following packages:
    conda create -n DeepMEns python=3.7
    conda activate DeepMEns
    pip install tensorFlow==2.4.1
    pip install keras==2.10.0
    pip install numpy==1.19.5
    pip install pandas==1.3.5
## Useage
- `5_Wt/HF1/esp_DeepMEns.py`: Python package, storing `DeepMEns` related codes in WT-SpCas9, eSpCas9(1.1), and SpCas9-HF1 datasets.

- `HF1_dataset, WT_dataset, esp_dataset, independent_test_datasets`: Contains all datasets and features for experiments.

- `Wt/HF1/esp_DeepMEns_0.h5, Wt/HF1/esp_DeepMEns_1.h5, Wt/HF1/esp_DeepMEns_2.h5, Wt/HF1/esp_DeepMEns_3.h5, Wt/HF1/esp_DeepMEns_4.h5,`: They are models trained by training sets in code that can be used directly in WT-SpCas9, eSpCas9(1.1), and SpCas9-HF1 datasets.
