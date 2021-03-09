REL_PATH=../../../
DIR_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")-$1"

python "${REL_PATH}monitor.py" "${EXP_DIR}" 1 &

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
--epochs=2 \
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

# pretrain exp-2021-0309-110608-VI_SP_VA_1080TI:
#  mean-top accs @ (min=15.810, mean=18.892, std=2.798) tensor([20.7200, 15.8100, 17.3000, 21.7400]))
#  best     accs @ (min=15.810, mean=18.892, std=2.798) tensor([20.7200, 15.8100, 17.3000, 21.7400]))

# pretrain exp-2021-0309-111354-VI_SP_VA_1080TI:
#  mean-top accs @ (min=15.810, mean=18.892, std=2.798) tensor([20.7200, 15.8100, 17.3000, 21.7400]))
#  best     accs @ (min=15.810, mean=18.892, std=2.798) tensor([20.7200, 15.8100, 17.3000, 21.7400]))

# pretrain exp-2021-0309-113737-VI_SP_VA_1080TI:
#  mean-top accs @ (min=15.810, mean=18.892, std=2.798) tensor([20.7200, 15.8100, 17.3000, 21.7400]))
#  best     accs @ (min=15.810, mean=18.892, std=2.798) tensor([20.7200, 15.8100, 17.3000, 21.7400]))

# pretrain exp-2021-0309-114707-VI_SP_VA_1080TI:
#  mean-top accs @ (min=15.810, mean=18.892, std=2.798) tensor([20.7200, 15.8100, 17.3000, 21.7400]))
#  best     accs @ (min=15.810, mean=18.892, std=2.798) tensor([20.7200, 15.8100, 17.3000, 21.7400]))

# pretrain exp-2021-0309-120327-VI_SP_VA_1080TI:
#  mean-top accs @ (min=15.810, mean=18.892, std=2.798) tensor([20.7200, 15.8100, 17.3000, 21.7400]))
#  best     accs @ (min=15.810, mean=18.892, std=2.798) tensor([20.7200, 15.8100, 17.3000, 21.7400]))

