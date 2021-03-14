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
--swap_idx=4 \
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



# pretrain exp-2021-0314-153459-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0., 0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.807, mean=88.499, std=0.176) tensor([88.2905, 88.8070, 88.5380, 88.4230, 88.4765, 88.3915, 88.7025, 88.3660]))
#  best     acc1s @ (max=89.010, mean=88.684, std=0.199) tensor([88.4800, 89.0100, 88.8000, 88.5400, 88.7400, 88.5100, 88.8700, 88.5200]))


# lnr_eval exp-2021-0314-153459-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.670, mean=99.631, std=0.029) tensor([99.6100, 99.6500, 99.5800, 99.6200, 99.6200, 99.6700, 99.6400, 99.6600]))
#  mean-top acc1s @ (max=89.192, mean=89.088, std=0.071) tensor([89.1140, 89.0640, 89.0120, 89.1680, 89.1920, 89.0040, 89.0340, 89.1160]))
#  best     acc1s @ (max=89.250, mean=89.134, std=0.072) tensor([89.1400, 89.1300, 89.0600, 89.2000, 89.2500, 89.0300, 89.1000, 89.1600]))

