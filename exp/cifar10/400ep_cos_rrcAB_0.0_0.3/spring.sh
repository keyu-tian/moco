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
--rrc_test=AB_0.0_0.3 \
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



# pretrain exp-2021-0406-044703-spring_scheduler-1080ti:
#  avg tr losses  tensor([6.1830, 6.1796, 6.1903, 6.1841])
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=86.993, mean=86.840, std=0.108) tensor([86.8365, 86.7455, 86.9925, 86.7860]))
#  best     acc1s @ (max=87.090, mean=86.935, std=0.114) tensor([86.9500, 86.8500, 87.0900, 86.8500]))


# lnr_eval exp-2021-0406-044703-spring_scheduler-1080ti:
#  avg tr losses  tensor([0.4629, 0.4993, 0.4908, 0.5109])
#  best     acc5s @ (max=99.730, mean=99.695, std=0.029) tensor([99.6900, 99.7000, 99.7300, 99.6600]))
#  mean-top acc1s @ (max=89.340, mean=88.900, std=0.339) tensor([88.6780, 88.9880, 89.3400, 88.5940]))
#  best     acc1s @ (max=89.370, mean=88.933, std=0.347) tensor([88.6900, 89.0500, 89.3700, 88.6200]))

