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
--swap_inv \
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



# pretrain exp-2021-0312-034509-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=81.680, mean=81.613, std=0.080) tensor([81.6135, 81.6800, 81.5000, 81.6570]))
#  best     acc1s @ (max=81.780, mean=81.698, std=0.075) tensor([81.6900, 81.7800, 81.6000, 81.7200]))


# lnr_eval exp-2021-0312-034509-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.280, mean=99.270, std=0.014) tensor([99.2800, 99.2800, 99.2500, 99.2700]))
#  mean-top acc1s @ (max=83.926, mean=83.890, std=0.036) tensor([83.9260, 83.8460, 83.9100, 83.8760]))
#  best     acc1s @ (max=83.990, mean=83.930, std=0.067) tensor([83.9900, 83.8500, 83.9800, 83.9000]))

