# PHMGNNBenchmark
## The emerging graph neural networks for intelligent fault diagnostics and prognostics: A guideline and a benchmark study
![PHMGNNBenchmark](https://github.com/HazeDT/PHMGNNBenchmark/blob/main/logo2.png)


# Implementation of the paper:
Paper:
```
@article{PHMGNNBenchmark,
  title={The emerging graph neural networks for intelligent fault diagnostics and prognostics: A guideline and a benchmark study},
  author={Tianfu Li and Zheng Zhou and Sinan Li  and Chuang Sun and Ruqiang Yan and Xuefeng Chen,},
  journal={Mechanical Systems and Signal Processing},
  Doi = {j.ymssp.2021.108653},
  year={2022}
}
```

![PHMGNNBenchmark](https://github.com/HazeDT/PHMGNNBenchmark/blob/main/Framework.png)

# Requirements
* Python 3.8 or newer
* torch-geometric 1.6.1
* pytorch  1.6.0
* pandas  1.0.5
* numpy  1.18.5

# Run the code
## For fault diagnostic
  * Node level fault daignostic <br>
  python  ./train_graph_diagnosis.py --model_name GCN --data_dir ./data/XJTU_Spurgear  --Input_type TD  --task Node   --checkpoint_dir ./checkpoint
  * Graph level fault daignostic <br>
  python  ./train_graph_diagnosis.py --model_name GCN --data_dir ./data/XJTU_Spurgear  --Input_type TD  --task Graph --pooltype EdgePool  --checkpoint_dir ./checkpoint
## For prognostic 
  python  ./train_graph_prognosis.py --model_name GCN --pooltype EdgePool --data_name CMAPSS_graph --data_file FD001 --data_dir ./data/CMAPSS --checkpoint_dir ./checkpoint/FD001
## The data for runing the demo
   In order to facilitate your implementation, we give some processed data here [Data for demo](https://drive.google.com/drive/folders/1px8KlGmWQ1SGkG-SKsw_j4tNDCsNI_38?usp=sharing).
   
# Datasets
## Self-collected datasets
* [XJTUSpurgear Dataset](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay?usp=sharing)
* [XJTUGearbox Dataset](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay?usp=sharing)
## Open source datasets
* [CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter)
* [MFPT Bearing Dataset](https://www.mfpt.org/fault-data-sets/)
* [PU Bearing Dataset](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/)
* [SEU Gearbox Dataset](https://github.com/cathysiyu/Mechanical-datasets)
