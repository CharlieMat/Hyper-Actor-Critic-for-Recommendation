mkdir -p output

# RL4RS environment

mkdir -p output/rl4rs/
mkdir -p output/rl4rs/env/
mkdir -p output/rl4rs/env/log/
mkdir -p output/rl4rs/agents/

output_path="output/rl4rs/"
log_name="rl4rs_user_env_lr0.001_reg0.0003"


N_ITER=50000
CONTINUE_ITER=0
GAMMA=0.9
TOPK=1
EMPTY=0

MAX_STEP=20
INITEP=0
REG=0.00001
NOISE=0.01
ELBOW=0.1
EP_BS=32
BS=64
SEED=17
SCORER="SASRec"
CRITIC_LR=0.001
ACTOR_LR=0.0001
BEHAVE_LR=0
TEMPER_SWEET_POINT=0.9

for NOISE in 0.1
do
    for TOPK in 1 # 0.8 0.6 0.4 0.2 0.1
    do
        for HA_COEF in 0.1
        do
            for BEHAVE_LR in 0 0.00001 0.0001
            do
                for ACTOR_LR in 0.0001
                do
                    for SEED in 7 # 11 13 17 19 23
                    do
                        mkdir -p ${output_path}agents/hac_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_hacoef${HA_COEF}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_topk${TOPK}_seed${SEED}/

                        python train_ddpg.py\
                            --env_class RL4RSEnvironment_GPU\
                            --policy_class ${SCORER}\
                            --critic_class GeneralCritic\
                            --agent_class HAC\
                            --facade_class OneStageFacade_HyperAction\
                            --seed ${SEED}\
                            --cuda 0\
                            --env_path ${output_path}env/${log_name}.env\
                            --max_step_per_episode ${MAX_STEP}\
                            --initial_temper ${MAX_STEP}\
                            --temper_sweet_point ${TEMPER_SWEET_POINT}\
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
                            --noise_var ${NOISE}\
                            --empty_start_rate ${EMPTY}\
                            --save_path ${output_path}agents/hac_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_hacoef${HA_COEF}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_topk${TOPK}_seed${SEED}/model\
                            --episode_batch_size ${EP_BS}\
                            --batch_size ${BS}\
                            --actor_lr ${ACTOR_LR}\
                            --critic_lr ${CRITIC_LR}\
                            --behavior_lr ${BEHAVE_LR}\
                            --hyper_actor_coef ${HA_COEF}\
                            --actor_decay ${REG}\
                            --critic_decay ${REG}\
                            --behavior_decay ${REG}\
                            --target_mitigate_coef 0.01\
                            --gamma ${GAMMA}\
                            --n_iter ${N_ITER}\
                            --initial_greedy_epsilon ${INITEP}\
                            --final_greedy_epsilon ${INITEP}\
                            --elbow_greedy ${ELBOW}\
                            --check_episode 10\
                            --topk_rate ${TOPK}\
                            > ${output_path}agents/hac_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_hacoef${HA_COEF}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_topk${TOPK}_seed${SEED}/log
                    done
                done
            done
        done
    done
done
