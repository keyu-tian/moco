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
--swap_epochs=400 \
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



# pretrain exp-2021-0312-034347-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.531, mean=88.235, std=0.240) tensor([87.9490, 88.2665, 88.1925, 88.5310]))
#  best     acc1s @ (max=88.610, mean=88.318, std=0.239) tensor([88.0300, 88.3500, 88.2800, 88.6100]))


# lnr_eval exp-2021-0312-034347-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.620, mean=99.600, std=0.018) tensor([99.6200, 99.5900, 99.6100, 99.5800]))
#  mean-top acc1s @ (max=89.098, mean=89.035, std=0.045) tensor([89.0360, 88.9920, 89.0160, 89.0980]))
#  best     acc1s @ (max=89.150, mean=89.115, std=0.045) tensor([89.1400, 89.0500, 89.1200, 89.1500]))

