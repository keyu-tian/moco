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
--swap_epochs=1 \
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



# pretrain exp-2021-0312-034551-VI_Face_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.727, mean=88.562, std=0.125) tensor([88.7265, 88.5675, 88.4265, 88.5275]))
#  best     acc1s @ (max=88.860, mean=88.725, std=0.099) tensor([88.8600, 88.7100, 88.7100, 88.6200]))


# lnr_eval exp-2021-0312-034551-VI_Face_1080TI:
#  best     acc5s @ (max=99.740, mean=99.707, std=0.028) tensor([99.6900, 99.7400, 99.7200, 99.6800]))
#  mean-top acc1s @ (max=89.428, mean=89.411, std=0.013) tensor([89.4160, 89.3980, 89.4280, 89.4040]))
#  best     acc1s @ (max=89.460, mean=89.450, std=0.008) tensor([89.4500, 89.4600, 89.4500, 89.4400]))

