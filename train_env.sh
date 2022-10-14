mkdir -p output

# RL4RS environment

mkdir -p output/rl4rs/
mkdir -p output/rl4rs/env
mkdir -p output/rl4rs/env/log

data_path="dataset/rl4rs/"
output_path="output/rl4rs/"

for REG in 0.0001 0.0003 0.001
do
    for LR in 0.0003 0.001 0.003
    do
        python train_env.py\
            --model RL4RSUserResponse\
            --reader RL4RSDataReader\
            --train_file ${data_path}rl4rs_b_train.csv\
            --val_file ${data_path}rl4rs_b_test.csv\
            --item_meta_file ${data_path}item_info.csv\
            --data_separator '@'\
            --meta_data_separator ' '\
            --loss 'bce'\
            --l2_coef ${REG}\
            --lr ${LR}\
            --epoch 2\
            --seed 19\
            --model_path ${output_path}env/rl4rs_user_env_lr${LR}_reg${REG}.model\
            --max_seq_len 50\
            --n_worker 4\
            --feature_dim 16\
            --hidden_dims 256\
            --attn_n_head 2\
            > ${output_path}env/log/rl4rs_user_env_lr${LR}_reg${REG}.model.log
    done
done