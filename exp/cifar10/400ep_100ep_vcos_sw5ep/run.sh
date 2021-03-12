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
--swap_epochs=5 \
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



# pretrain exp-2021-0312-030212-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.694, mean=88.583, std=0.121) tensor([88.6805, 88.4610, 88.4985, 88.6940]))
#  best     acc1s @ (max=88.810, mean=88.688, std=0.132) tensor([88.7600, 88.5100, 88.6700, 88.8100]))


# lnr_eval exp-2021-0312-030212-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.760, mean=99.730, std=0.022) tensor([99.7100, 99.7600, 99.7300, 99.7200]))
#  mean-top acc1s @ (max=89.628, mean=89.562, std=0.045) tensor([89.6280, 89.5280, 89.5480, 89.5420]))
#  best     acc1s @ (max=89.650, mean=89.610, std=0.037) tensor([89.6500, 89.5600, 89.6100, 89.6200]))

