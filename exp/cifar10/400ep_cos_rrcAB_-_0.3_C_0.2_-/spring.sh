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
--rrc_test=AB_-_0.3_C_0.2_- \
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



# pretrain exp-2021-0406-044956-spring_scheduler-1080ti:
#  avg tr losses  tensor([5.1189, 5.1224, 5.1253, 5.1319])
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.103, mean=88.018, std=0.090) tensor([88.0055, 88.1030, 88.0660, 87.8975]))
#  best     acc1s @ (max=88.150, mean=88.103, std=0.071) tensor([88.1100, 88.1500, 88.1500, 88.0000]))


# lnr_eval exp-2021-0406-044956-spring_scheduler-1080ti:
#  avg tr losses  tensor([0.4553, 0.6287, 0.4343, 0.4645])
#  best     acc5s @ (max=99.660, mean=99.632, std=0.019) tensor([99.6200, 99.6300, 99.6600, 99.6200]))
#  mean-top acc1s @ (max=89.622, mean=89.340, std=0.201) tensor([89.2940, 89.6220, 89.3000, 89.1460]))
#  best     acc1s @ (max=89.640, mean=89.397, std=0.183) tensor([89.3500, 89.6400, 89.4000, 89.2000]))

