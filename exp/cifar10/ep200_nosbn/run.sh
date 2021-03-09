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
#--sbn \
#--warmup
#--nowd
#--mlp

#--resume_ckpt=

failed=$?

RESULT=$(tail "${EXP_DIR}"/log.txt -n 1)
echo ""
echo -e "\033[36mat ${PWD#}/${EXP_DIR}\033[0m"
echo -e "\033[36m${RESULT#*@}\033[0m"

#fg
if [ $failed -ne 0 ]; then
    sh "./kill.sh"
else
    touch "${EXP_DIR}".terminate
fi

