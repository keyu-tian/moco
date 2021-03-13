REL_PATH=../../../
DIR_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")-$1"

python "${REL_PATH}monitor.py" "${EXP_DIR}" &

PYTHONPATH=${PYTHONPATH}:${REL_PATH} \
srun \
--job-name "${DIR_NAME}----${EXP_DIR}" \
--mpi=pmi2 -p $1 -n4 --gres=gpu:4 \
--ntasks-per-node=4 \
--cpus-per-task=5 \
python -u -m main_cifar \
--main_py_rel_path="${REL_PATH}" \
--exp_dirname="${EXP_DIR}" \
--moco_m=0.99 \
--moco_t=0.1 \
--moco_symm \
--epochs=200 \
--coslr \
--dataset=cifar10 \
--num_workers=4 \
--pin_mem \
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

# with init:
# pretrain exp-2021-0309-120416-VI_SP_VA_1080TI:
#  mean-top accs @ (min=83.829, mean=84.087, std=0.242) tensor([84.4080, 84.0120, 84.1000, 83.8290]))
#  best     accs @ (min=83.920, mean=84.162, std=0.236) tensor([84.4800, 84.0800, 84.1700, 83.9200]))

# without init:
# pretrain exp-2021-0309-210701-VI_SP_VA_1080TI:
#  mean-top accs @ (max=85.626, mean=85.458, std=0.184) tensor([85.2380, 85.3770, 85.6260, 85.5920]))
#  best     accs @ (max=85.760, mean=85.560, std=0.205) tensor([85.3100, 85.4800, 85.6900, 85.7600]))

