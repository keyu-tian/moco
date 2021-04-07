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
--rrc_test=Rand500w \
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



# pretrain exp-2021-0406-045154-spring_scheduler-1080ti:
#  avg tr losses  tensor([4.0812, 4.0740, 4.0787, 4.0755])
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.397, mean=88.271, std=0.123) tensor([88.1145, 88.2385, 88.3320, 88.3970]))
#  best     acc1s @ (max=88.510, mean=88.383, std=0.133) tensor([88.2200, 88.3300, 88.4700, 88.5100]))


# lnr_eval exp-2021-0406-045154-spring_scheduler-1080ti:
#  avg tr losses  tensor([0.6207, 0.7816, 0.4978, 0.5294])
#  best     acc5s @ (max=99.710, mean=99.690, std=0.022) tensor([99.6900, 99.7100, 99.7000, 99.6600]))
#  mean-top acc1s @ (max=89.178, mean=89.036, std=0.225) tensor([89.1620, 88.7020, 89.1000, 89.1780]))
#  best     acc1s @ (max=89.310, mean=89.105, std=0.228) tensor([89.1900, 88.7800, 89.1400, 89.3100]))

