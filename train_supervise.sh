mkdir -p output

# RL4RS environment

mkdir -p output/rl4rs/
mkdir -p output/rl4rs/env/
mkdir -p output/rl4rs/env/log/
mkdir -p output/rl4rs/agents/

output_path="output/rl4rs/"
log_name="rl4rs_user_env_lr0.001_reg0.0003"


N_ITER=80000
CONTINUE_ITER=0
GAMMA=0.9
TOPK=1
EMPTY=0

MAX_STEP=20
INITEP=0
REG=0.00003
NOISE=0.1
ELBOW=0.1
EP_BS=32
BS=64
SEED=17
SCORER="SASRec"
CRITIC_LR=0.001
ACTOR_LR=0.0001
BEHAVE_LR=0
TEMPER_RATE=1.0

for REG in 0.00001
do
    for ACTOR_LR in 0.0001 0.0005 0.00005 0.00001
    do
        for SEED in 11 # 13 17 19 23
        do
            mkdir -p ${output_path}agents/offline_${SCORER}_actor${ACTOR_LR}_niter${N_ITER}_reg${REG}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}/

            python train_ddpg.py\
                --env_class RL4RSEnvironment_GPU\
                --policy_class ${SCORER}\
                --critic_class GeneralCritic\
                --agent_class OfflineSLAgent\
                --facade_class OneStageFacade\
                --seed ${SEED}\
                --cuda 0\
                --env_path ${output_path}env/${log_name}.env\
                --max_step_per_episode ${MAX_STEP}\
                --initial_temper ${MAX_STEP}\
                --reward_func mean_with_cost\
                --urm_log_path ${output_path}env/log/${log_name}.model.log\
                --sasrec_n_layer 2\
                --sasrec_d_model 32\
                --sasrec_n_head 4\
                --sasrec_dropout 0.1\
                --critic_hidden_dims 256 64\
                --slate_size 9\
                --buffer_size 100000\
                --start_timestamp 2000\
                --empty_start_rate ${EMPTY}\
                --save_path ${output_path}agents/offline_${SCORER}_actor${ACTOR_LR}_niter${N_ITER}_reg${REG}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}/model\
                --episode_batch_size ${EP_BS}\
                --batch_size ${BS}\
                --actor_lr ${ACTOR_LR}\
                --actor_decay ${REG}\
                --n_iter ${N_ITER}\
                --check_episode 10\
                --topk_rate ${TOPK}\
                > ${output_path}agents/offline_${SCORER}_actor${ACTOR_LR}_niter${N_ITER}_reg${REG}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}/log
        done
    done
done
