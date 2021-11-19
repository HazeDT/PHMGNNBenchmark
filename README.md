# PHMGNNBenchmark
## The emerging graph neural networks for intelligent fault diagnostics and prognostics: A guideline and a benchmark study
![PHMGNNBenchmark](https://github.com/HazeDT/PHMGNNBenchmark/blob/main/logo2.png)


# Implementation of the paper:
Tianfu Li, Zheng Zhou, Sinan Li, Chuang Sun, Ruqiang Yan, Xuefeng Chen, “The emerging graph neural networks for intelligent fault diagnostics and prognostics: A guideline and a benchmark study,” Mechanical Systems and Signal Processing, 2021. DOI: j.ymssp.2021.108653
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
  


