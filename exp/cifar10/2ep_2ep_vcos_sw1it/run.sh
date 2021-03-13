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
--epochs=2 \
--coslr \
--eval_epochs=2 \
--eval_coslr \
--dataset=cifar10 \
--num_workers=4 \
--pin_mem \
--swap_iters=1 \
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



# pretrain exp-2021-0312-024949-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=37.760, mean=37.345, std=0.451) tensor([37.6300, 36.7600, 37.7600, 37.2300]))
#  best     acc1s @ (max=37.760, mean=37.345, std=0.451) tensor([37.6300, 36.7600, 37.7600, 37.2300]))


# lnr_eval exp-2021-0312-024949-VI_SP_VA_1080TI:
#  best     acc5s @ (max=83.740, mean=82.850, std=0.685) tensor([82.6600, 82.0900, 83.7400, 82.9100]))
#  mean-top acc1s @ (max=37.760, mean=37.140, std=0.427) tensor([36.7800, 37.0200, 37.7600, 37.0000]))
#  best     acc1s @ (max=37.760, mean=37.140, std=0.427) tensor([36.7800, 37.0200, 37.7600, 37.0000]))

