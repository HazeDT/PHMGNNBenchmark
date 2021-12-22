# PHMGNNBenchmark
## [The emerging graph neural networks for intelligent fault diagnostics and prognostics: A guideline and a benchmark study](https://www.sciencedirect.com/science/article/pii/S0888327021009791)
![PHMGNNBenchmark](https://github.com/HazeDT/PHMGNNBenchmark/blob/main/logo2.png)


# Implementation of the paper:
Paper:
```
@article{PHMGNNBenchmark,
  title={The emerging graph neural networks for intelligent fault diagnostics and prognostics: A guideline and a benchmark study},
  author = {Tianfu Li and Zheng Zhou and Sinan Li and Chuang Sun and Ruqiang Yan and Xuefeng Chen},
  journal={Mechanical Systems and Signal Processing},
  volume = {168},
  pages = {108653},
  year = {2022},
  issn = {0888-3270},
  doi = {https://doi.org/10.1016/j.ymssp.2021.108653},
  url = {https://www.sciencedirect.com/science/article/pii/S0888327021009791},
}
```

![PHMGNNBenchmark](https://github.com/HazeDT/PHMGNNBenchmark/blob/main/Framework.png)

# Requirements
* Python 3.8 or newer
* torch-geometric 1.6.1
* pytorch  1.6.0
* pandas  1.0.5
* numpy  1.18.5

# Guide 
 We provide a novel intelligent fault diagnostics and prognostics framework based on GNNs. The framework consists of two branches, that is, the node-level fault diagnostics architecture and graph-level fault diagnostics or regression architecture. In node-level fault diagnosis, each node of a graph is considered as a sample, while the entire graph is considered as a sample in graph-level fault diagnosis. <br> In this code library, we provide three graph constrcution methods (`KnnGraph`, `RadiusGraph`, and `PathGraph`), and two different input types (`Frequency domain` and `time domain`). Besides, seven GNNs and four graph pooling methods are implemented. 
 
# Pakages
* `datasets` contians the data load method for different dataset
* `model` contians the implemented model for nodel-level task
* `model2` contians the implemented model for graph-level rask

# Run the code
## For fault diagnostic
  * Node level fault daignostic <br>
  python  ./train_graph_diagnosis.py --model_name GCN --data_name XJTUGearboxRadius --data_dir ./data/XJTUGearbox/XJTUGearboxRadius.pkl  --Input_type TD  --task Node   --checkpoint_dir ./checkpoint 
  * Graph level fault daignostic <br>
  python  ./train_graph_diagnosis.py --model_name GCN --data_name XJTUGearboxRadius --data_dir ./data/XJTUGearbox --Input_type TD  --task Graph --pooltype EdgePool  --checkpoint_dir ./checkpoint
## For prognostic 
  python  ./train_graph_prognosis.py --model_name GCN --pooltype EdgePool --data_name CMAPSS_graph --data_file FD001 --data_dir ./data/CMAPSS/ --checkpoint_dir ./checkpoint/FD001
## The data for runing the demo
   In order to facilitate your implementation, we give some processed data here for node level-fault diagnosis and graph-level prognosis [`Data for demo`](https://drive.google.com/drive/folders/1px8KlGmWQ1SGkG-SKsw_j4tNDCsNI_38?usp=sharing).
   
# Datasets
## Fault diagnostic datasets
### Self-collected datasets
* [XJTUSpurgear Dataset](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay?usp=sharing)
* [XJTUGearbox Dataset](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay?usp=sharing)
### Open source datasets
* [CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter)
* [MFPT Bearing Dataset](https://www.mfpt.org/fault-data-sets/)
* [PU Bearing Dataset](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/)
* [SEU Gearbox Dataset](https://github.com/cathysiyu/Mechanical-datasets)
## Prognostic datasets
* [CMAPSS dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
* [PHM2010 dataset](https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/)

# Note
This code library is run under the `windows operating system`. If you run under the `linux operating system`, you need to delete the `‘/tmp’` before the path in the `dataset` to avoid path errors.
# Related works
* [Li et al. Multireceptive Field Graph Convolutional Networks for Machine Fault Diagnosis, IEEE Transactions on Industrial Electronics](https://ieeexplore.ieee.org/abstract/document/9280401)
* [Li et al. Domain Adversarial Graph Convolutional Network for Fault Diagnosis Under Variable Working Conditions, IEEE Transactions on Instrumentation and Measurement ](https://ieeexplore.ieee.org/abstract/document/9410617)
* [Li et al. Hierarchical attention graph convolutional network to fuse multi-sensor signals for remaining useful life prediction, Reliability Engineering & System Safety](https://www.sciencedirect.com/science/article/abs/pii/S0951832021003975)
