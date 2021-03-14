sleep "${2:-"1"}"

REL_PATH=../../../
DIR_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")-$1"

python "${REL_PATH}monitor.py" "${EXP_DIR}" &

PYTHONPATH=${PYTHONPATH}:${REL_PATH} \
srun \
--job-name "${DIR_NAME}----${EXP_DIR}" \
--mpi=pmi2 -p $1 -n8 --gres=gpu:8 \
--ntasks-per-node=8 \
--cpus-per-task=6 \
python -u -m main_cifar \
--main_py_rel_path="${REL_PATH}" \
--exp_dirname="${EXP_DIR}" \
--moco_m=0.99 \
--moco_t=0.1 \
--moco_symm \
--epochs=400 \
--coslr \
--warmup \
--grad_clip=None \
--eval_epochs=100 \
--eval_coslr \
--dataset=cifar10 \
--num_workers=4 \
--pin_mem \
--pret_verbose \
--swap_epochs=5 \
--swap_idx=5 \
#--reset_op \
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



# pretrain exp-2021-0314-153456-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0., 0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.776, mean=88.631, std=0.123) tensor([88.7760, 88.6925, 88.6780, 88.4800, 88.6825, 88.6905, 88.6430, 88.4080]))
#  best     acc1s @ (max=88.880, mean=88.755, std=0.110) tensor([88.8800, 88.8100, 88.7800, 88.5900, 88.7800, 88.8300, 88.7900, 88.5800]))


# lnr_eval exp-2021-0314-153456-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.730, mean=99.701, std=0.023) tensor([99.6700, 99.7300, 99.6900, 99.6900, 99.7200, 99.7300, 99.6800, 99.7000]))
#  mean-top acc1s @ (max=89.434, mean=89.375, std=0.056) tensor([89.3820, 89.3780, 89.3320, 89.2600, 89.4220, 89.4340, 89.3840, 89.4100]))
#  best     acc1s @ (max=89.450, mean=89.400, std=0.057) tensor([89.4200, 89.4000, 89.3600, 89.2800, 89.4500, 89.4500, 89.4100, 89.4300]))

