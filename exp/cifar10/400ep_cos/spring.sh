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
--warmup \
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



# pretrain exp-2021-0416-044527-:
#  avg tr losses  tensor([4.0727, 4.0753, 4.0761, 4.0660])
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.547, mean=88.334, std=0.172) tensor([88.5465, 88.3205, 88.1270, 88.3435]))
#  best     acc1s @ (max=88.610, mean=88.428, std=0.161) tensor([88.6100, 88.4600, 88.2200, 88.4200]))


# lnr_eval exp-2021-0416-044527-:
#  avg tr losses  tensor([0.5241, 0.6947, 0.5777, 0.8161])
#  best     acc5s @ (max=99.720, mean=99.665, std=0.047) tensor([99.6500, 99.6100, 99.6800, 99.7200]))
#  mean-top acc1s @ (max=89.182, mean=89.043, std=0.155) tensor([89.1180, 89.0480, 88.8260, 89.1820]))
#  best     acc1s @ (max=89.230, mean=89.097, std=0.170) tensor([89.2100, 89.0900, 88.8600, 89.2300]))

