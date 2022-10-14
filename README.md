## 0. Setup

```
conda create -n 2stageddpg python=3.9
conda activate 2stageddpg
conda install pytorch torchvision -c pytorch
conda install pandas matplotlib scikit-learn
pip install tqdm
conda install -c anaconda ipykernel
python -m ipykernel install --user --name 2stageddpg --display-name "2StageDDPG"
```

## 1. Data Preparation

For RL4RS data preparation, run cells in RL4RSData.ipynb. 

For ML1M data preparation, run cells in ML1MData.ipynb. 

## 2. Pretrain User Response Model as Environment Component

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

## 3. Training

#### 3.1 Script list

DDPG:
> bash train_ddpg.sh

BehaviorDDPG:
> bash train_superddpg.sh

Online Supervise Learning:
> bash train_online_sasrec.sh

Offline Supervise Learning:
> train_supervise.sh

Two-stage DDPG:
> bash train_ddpg_twostage.sh

#### 3.2 Continue training

Use the same script but change "--n_iter ${N_ITER}" to "--n_iter ${PREVIOUS_N_ITER} ${N_ITER}"

## 4. Result Observation

* BehaviorDDPGTraining.ipynb
* OfflineSL.ipynb
