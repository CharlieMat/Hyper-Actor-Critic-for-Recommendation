# Exploration and Regularization of the Latent Action Space in Recommendation

This repository is the implementation of [Exploration and Regularization of the Latent Action Space in Recommendation](https://arxiv.org/abs/2302.03431) in WWW 23'.

## Citing

```
@inproceedings{liu2023exploration,
  author = {Liu, Shuchang and Cai, Qingpeng and Sun, Bowen and Wang, Yuhao and Jiang, Ji and Zheng, Dong and Jiang, Peng and Gai, Kun and Zhao, Xiangyu and Zhang, Yongfeng},
  title = {Exploration and Regularization of the Latent Action Space in Recommendation},
  year = {2023},
  isbn = {9781450394161},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3543507.3583244},
  doi = {10.1145/3543507.3583244},
  pages = {833–844},
  numpages = {12},
  location = {Austin, TX, USA},
  series = {WWW '23}
}
```

## 0. Setup

```
conda create -n hac python=3.9
conda activate hac
conda install pytorch torchvision -c pytorch
conda install pandas matplotlib scikit-learn
pip install tqdm
conda install -c anaconda ipykernel
python -m ipykernel install --user --name hac --display-name "HAC"
```


## 1. Pretrain User Response Model as Environment Component

Modify train_env.sh:
* Change the directories, data_path, and output_path for your dataset
* Set the following arguments with X in {RL4RS, ML1M}:
  * --model {X}UserResponse\
  * --reader {X}DataReader\
  * --train_file ${data_path}{X}_b_train.csv\
  * --val_file ${data_path}{X}_b_test.csv\
* Set your model_path and log_path in the script.

Run:
> bash train_enb.sh

## 2. Training

#### 2.1 Script list

> bash train_xxx.sh

Examples:

DDPG:
> bash train_ddpg.sh

BehaviorDDPG:
> bash train_superddpg.sh

Online Supervise Learning:
> bash train_online_sasrec.sh

Offline Supervise Learning:
> train_supervise.sh

#### 2.2 Continue training

Use the same script but change "--n_iter ${N_ITER}" to "--n_iter ${PREVIOUS_N_ITER} ${N_ITER}"

## 3. Result Observation

> bash test.sh

* HACTraining.ipynb

