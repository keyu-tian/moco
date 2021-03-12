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
--swap_epochs=50 \
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



# pretrain exp-2021-0312-034339-VI_Face_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.346, mean=88.193, std=0.202) tensor([88.3220, 88.1980, 88.3460, 87.9070]))
#  best     acc1s @ (max=88.580, mean=88.393, std=0.204) tensor([88.5100, 88.3600, 88.5800, 88.1200]))


# lnr_eval exp-2021-0312-034339-VI_Face_1080TI:
#  best     acc5s @ (max=99.680, mean=99.658, std=0.032) tensor([99.6700, 99.6800, 99.6100, 99.6700]))
#  mean-top acc1s @ (max=89.316, mean=89.253, std=0.047) tensor([89.3160, 89.2360, 89.2040, 89.2540]))
#  best     acc1s @ (max=89.330, mean=89.267, std=0.046) tensor([89.3300, 89.2500, 89.2200, 89.2700]))

