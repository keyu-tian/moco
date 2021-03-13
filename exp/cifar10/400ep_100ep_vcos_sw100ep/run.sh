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
--epochs=400 \
--coslr \
--eval_epochs=100 \
--eval_coslr \
--dataset=cifar10 \
--num_workers=4 \
--pin_mem \
--swap_epochs=100 \
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



# pretrain exp-2021-0312-034512-VI_Face_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=87.505, mean=87.400, std=0.106) tensor([87.2710, 87.5045, 87.3590, 87.4645]))
#  best     acc1s @ (max=87.910, mean=87.728, std=0.171) tensor([87.5100, 87.8000, 87.6900, 87.9100]))


# lnr_eval exp-2021-0312-034512-VI_Face_1080TI:
#  best     acc5s @ (max=99.760, mean=99.747, std=0.010) tensor([99.7600, 99.7400, 99.7500, 99.7400]))
#  mean-top acc1s @ (max=89.052, mean=89.029, std=0.030) tensor([89.0460, 88.9860, 89.0300, 89.0520]))
#  best     acc1s @ (max=89.090, mean=89.065, std=0.031) tensor([89.0700, 89.0200, 89.0900, 89.0800]))

