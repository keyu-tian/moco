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
--rrc_test=AB_-_0.3 \
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



# pretrain exp-2021-0406-044817-spring_scheduler-1080ti:
#  avg tr losses  tensor([6.4296, 6.4226, 6.4273, 6.4201])
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=86.605, mean=86.430, std=0.167) tensor([86.4920, 86.4120, 86.2090, 86.6055]))
#  best     acc1s @ (max=86.680, mean=86.510, std=0.173) tensor([86.5700, 86.5200, 86.2700, 86.6800]))


# lnr_eval exp-2021-0406-044817-spring_scheduler-1080ti:
#  avg tr losses  tensor([0.4651, 0.4779, 0.5411, 0.5409])
#  best     acc5s @ (max=99.720, mean=99.680, std=0.039) tensor([99.6700, 99.7200, 99.6300, 99.7000]))
#  mean-top acc1s @ (max=88.908, mean=88.688, std=0.162) tensor([88.6880, 88.9080, 88.5240, 88.6320]))
#  best     acc1s @ (max=88.990, mean=88.757, std=0.181) tensor([88.7600, 88.9900, 88.5500, 88.7300]))

