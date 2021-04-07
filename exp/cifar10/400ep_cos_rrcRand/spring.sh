sleep "${2:-"0"}"

REL_PATH=../../../
DIR_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")-$1"

python "${REL_PATH}monitor.py" "${EXP_DIR}" &

PYTHONPATH=${PYTHONPATH}:${REL_PATH} \
srun \
--job-name "${DIR_NAME}----${EXP_DIR}" \
--mpi=pmi2 -p $1 --comment=spring-submit -n4 --gres=gpu:4 \
--ntasks-per-node=4 \
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
#--rrc_test=Rand30w \
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



# pretrain exp-2021-0406-045021-spring_scheduler-1080ti:
#  avg tr losses  tensor([4.0781, 4.0683, 4.0696, 4.0698])
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.456, mean=88.245, std=0.198) tensor([88.4560, 88.0810, 88.0705, 88.3715]))
#  best     acc1s @ (max=88.530, mean=88.355, std=0.191) tensor([88.5100, 88.2000, 88.1800, 88.5300]))


# lnr_eval exp-2021-0406-045021-spring_scheduler-1080ti:
#  avg tr losses  tensor([0.5055, 0.5349, 0.5026, 0.6396])
#  best     acc5s @ (max=99.720, mean=99.653, std=0.051) tensor([99.6300, 99.6000, 99.6600, 99.7200]))
#  mean-top acc1s @ (max=89.398, mean=89.085, std=0.236) tensor([89.3980, 88.9100, 88.8940, 89.1360]))
#  best     acc1s @ (max=89.570, mean=89.143, std=0.310) tensor([89.5700, 88.9300, 88.9000, 89.1700]))

