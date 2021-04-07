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
--rrc_test=AB_0.3_- \
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



# pretrain exp-2021-0406-044839-spring_scheduler-1080ti:
#  avg tr losses  tensor([2.1974, 2.2049, 2.1966, 2.1985])
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=86.747, mean=86.577, std=0.170) tensor([86.6345, 86.3445, 86.5810, 86.7470]))
#  best     acc1s @ (max=86.900, mean=86.670, std=0.216) tensor([86.6700, 86.3800, 86.7300, 86.9000]))


# lnr_eval exp-2021-0406-044839-spring_scheduler-1080ti:
#  avg tr losses  tensor([0.8504, 0.9423, 0.6560, 0.7990])
#  best     acc5s @ (max=99.510, mean=99.452, std=0.051) tensor([99.3900, 99.5100, 99.4700, 99.4400]))
#  mean-top acc1s @ (max=87.108, mean=86.829, std=0.274) tensor([87.1080, 86.5140, 86.6940, 87.0000]))
#  best     acc1s @ (max=87.180, mean=86.878, std=0.267) tensor([87.1800, 86.5900, 86.7300, 87.0100]))

