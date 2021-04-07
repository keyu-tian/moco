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
--rrc_test=C_0.2_- \
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



# pretrain exp-2021-0406-043952-spring_scheduler-1080ti:
#  avg tr losses  tensor([3.1401, 3.1444, 3.1530, 3.1363])
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.090, mean=87.948, std=0.186) tensor([88.0385, 87.9865, 88.0905, 87.6755]))
#  best     acc1s @ (max=88.210, mean=88.087, std=0.193) tensor([88.1700, 88.1700, 88.2100, 87.8000]))


# lnr_eval exp-2021-0406-043952-spring_scheduler-1080ti:
#  avg tr losses  tensor([0.8257, 0.5104, 0.5698, 0.5484])
#  best     acc5s @ (max=99.680, mean=99.607, std=0.050) tensor([99.6800, 99.5700, 99.6000, 99.5800]))
#  mean-top acc1s @ (max=88.786, mean=88.583, std=0.210) tensor([88.7860, 88.7100, 88.5180, 88.3180]))
#  best     acc1s @ (max=88.820, mean=88.630, std=0.214) tensor([88.8200, 88.7900, 88.5400, 88.3700]))

