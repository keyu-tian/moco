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
--rrc_test=AB_0.0_- \
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



# pretrain exp-2021-0406-044640-spring_scheduler-1080ti:
#  avg tr losses  tensor([3.9099, 3.9207, 3.9172, 3.9127])
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.298, mean=88.162, std=0.145) tensor([88.1090, 88.2595, 88.2980, 87.9820]))
#  best     acc1s @ (max=88.390, mean=88.285, std=0.137) tensor([88.2900, 88.3700, 88.3900, 88.0900]))


# lnr_eval exp-2021-0406-044640-spring_scheduler-1080ti:
#  avg tr losses  tensor([0.4999, 0.5694, 0.6121, 0.5038])
#  best     acc5s @ (max=99.670, mean=99.640, std=0.032) tensor([99.6600, 99.6300, 99.6700, 99.6000]))
#  mean-top acc1s @ (max=89.366, mean=89.193, std=0.190) tensor([89.3660, 89.0060, 89.0540, 89.3480]))
#  best     acc1s @ (max=89.420, mean=89.243, std=0.199) tensor([89.4200, 89.0600, 89.0800, 89.4100]))

