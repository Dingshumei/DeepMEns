# DeepMEns: an ensemble model for predicting of sgRNA on-target activity based on multiple features
## Environment
-The algorithm mainly uses the following packages:
conda create -n DeepMEns python=3.7
conda activate DeepMEns
pip install tensorFlow==2.4.1
pip install keras==2.10.0
pip install numpy==1.19.5
pip install pandas==1.3.5
## Useage
- 5_Wt/HF1/esp_DeepMEns.py: Python package, storing `DeepMEns` related codes in WT-SpCas9, eSpCas9(1.1), and SpCas9-HF1 datasets.
- HF1_dataset, WT_dataset, esp_dataset, independent_test_datasets: Contains all datasets and features for experiments.
- Wt/HF1/esp_DeepMEns_0.h5, Wt/HF1/esp_DeepMEns_1.h5, Wt/HF1/esp_DeepMEns_2.h5, Wt/HF1/esp_DeepMEns_3.h5, Wt/HF1/esp_DeepMEns_4.h5: They are models trained by training sets in code that can be used directly in WT-SpCas9, eSpCas9(1.1), and SpCas9-HF1 datasets.
- hand-crafted-feature.py: Code that contains the biological characteristics used in the article.
