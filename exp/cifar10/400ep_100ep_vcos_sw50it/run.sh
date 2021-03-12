REL_PATH=../../../
DIR_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")-$1"

python "${REL_PATH}monitor.py" "${EXP_DIR}" &

PYTHONPATH=${PYTHONPATH}:${REL_PATH} \
srun \
--job-name "${DIR_NAME}" \
--mpi=pmi2 -p $1 -n4 --gres=gpu:4 \
--ntasks-per-node=4 \
--cpus-per-task=5 \
python -u -m main_cifar \
--main_py_rel_path="${REL_PATH}" \
--exp_dirname="${EXP_DIR}" \
--moco_m=0.99 \
--moco_t=0.1 \
--moco_symm \
--epochs=400 \
--coslr \
--eval_epochs=100 \
--eval_coslr \
--dataset=cifar10 \
--num_workers=4 \
--pin_mem \
--swap_iters=50 \
#--sbn \
#--warmup
#--nowd
#--mlp
#--seed_base=0 \

#--resume_ckpt=

failed=$?
echo "failed=${failed}"

RESULT=$(tail "${EXP_DIR}"/log.txt -n 1)
echo ""
echo -e "\033[36mat ${PWD#}/${EXP_DIR}\033[0m"
echo -e "\033[36m${RESULT#*@}\033[0m"

#fg
if [ $failed -ne 0 ]; then
  sh "./kill.sh"
  echo "killed."
else
  touch "${EXP_DIR}".terminate
fi



# pretrain exp-2021-0312-040933-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.518, mean=88.416, std=0.109) tensor([88.5175, 88.2660, 88.4085, 88.4705]))
#  best     acc1s @ (max=88.650, mean=88.542, std=0.109) tensor([88.6000, 88.4000, 88.5200, 88.6500]))


# lnr_eval exp-2021-0312-040933-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.700, mean=99.688, std=0.019) tensor([99.6900, 99.7000, 99.6600, 99.7000]))
#  mean-top acc1s @ (max=89.474, mean=89.419, std=0.037) tensor([89.3980, 89.4040, 89.4740, 89.3980]))
#  best     acc1s @ (max=89.510, mean=89.490, std=0.024) tensor([89.5100, 89.4600, 89.5100, 89.4800]))

