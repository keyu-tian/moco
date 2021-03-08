REL_PATH=../../../
DIR_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")-$1"
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
--sbn \
--moco_symm \
--epochs=2 \
--coslr \
--dataset=cifar10 \
--num_workers=4 \
--pin_mem \

RESULT=$(tail "${EXP_DIR}"/log.txt -n 1)
echo ""
echo -e "\033[36mat ${PWD#}/${EXP_DIR}\033[0m"
echo -e "\033[36m${RESULT#*@}\033[0m"

cd "${EXP_DIR}" || exit
python "${REL_PATH}../show.py"


#--warmup
#--nowd
#--mlp

#--resume_ckpt=
