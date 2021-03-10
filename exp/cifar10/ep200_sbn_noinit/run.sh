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
--seed_base=0 \
--moco_m=0.99 \
--moco_t=0.1 \
--moco_symm \
--epochs=200 \
--coslr \
--dataset=cifar10 \
--num_workers=4 \
--pin_mem \
--sbn \
#--warmup
#--nowd
#--mlp

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

# with init:
# pretrain exp-2021-0309-120443-VI_SP_VA_1080TI:
#  mean-top accs @ (min=83.624, mean=83.864, std=0.250) tensor([84.2160, 83.8170, 83.8000, 83.6240]))
#  best     accs @ (min=83.690, mean=83.943, std=0.248) tensor([84.2800, 83.9400, 83.8600, 83.6900]))

# without init:# pretrain exp-2021-0309-204111-VI_SP_VA_1080TI:
#  mean-top accs @ (max=85.366, mean=84.990, std=0.270) tensor([84.8440, 85.0000, 84.7520, 85.3660]))
#  best     accs @ (max=85.410, mean=85.097, std=0.231) tensor([84.9500, 85.1300, 84.9000, 85.4100]))

