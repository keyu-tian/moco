sleep "${2:-"0"}"

REL_PATH=../../../
DIR_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")-$1"

python "${REL_PATH}monitor.py" "${EXP_DIR}" &

PYTHONPATH=${PYTHONPATH}:${REL_PATH} \
srun \
--job-name "${DIR_NAME}----${EXP_DIR}" \
--mpi=pmi2 -p $1 --comment=spring-submit -n8 --gres=gpu:8 \
--ntasks-per-node=8 \
--cpus-per-task=6 \
python -u -m main_cifar \
--main_py_rel_path="${REL_PATH}" \
--exp_dirname="${EXP_DIR}" \
--dataset=cifar10 \
--ds_root=None \
--moco_m=0.99 \
--moco_t=0.1 \
--moco_symm \
--epochs=400 \
--coslr \
--eval_epochs=100 \
--eval_coslr \
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



# pretrain exp-2021-0416-044551-:
#  avg tr losses  tensor([4.0876, 4.0972, 4.1036, 4.0937])
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.550, mean=88.434, std=0.121) tensor([88.3050, 88.5500, 88.3575, 88.5250]))
#  best     acc1s @ (max=88.650, mean=88.535, std=0.111) tensor([88.4500, 88.6500, 88.4300, 88.6100]))


# lnr_eval exp-2021-0416-044551-:
#  avg tr losses  tensor([0.6400, 0.5761, 0.7223, 0.5530])
#  best     acc5s @ (max=99.720, mean=99.655, std=0.047) tensor([99.7200, 99.6200, 99.6600, 99.6200]))
#  mean-top acc1s @ (max=89.296, mean=89.118, std=0.153) tensor([88.9240, 89.2960, 89.1440, 89.1080]))
#  best     acc1s @ (max=89.310, mean=89.175, std=0.139) tensor([88.9800, 89.3100, 89.2000, 89.2100]))

