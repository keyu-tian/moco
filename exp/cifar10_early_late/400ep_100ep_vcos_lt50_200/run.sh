sleep "${2:-"1"}"

REL_PATH=../../../
DIR_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")-$1"

python "${REL_PATH}monitor.py" "${EXP_DIR}" &

PYTHONPATH=${PYTHONPATH}:${REL_PATH} \
srun \
--job-name "${DIR_NAME}----${EXP_DIR}" \
--mpi=pmi2 -p $1 -n8 --gres=gpu:8 \
--ntasks-per-node=8 \
--cpus-per-task=6 \
python -u -m main_cifar \
--main_py_rel_path="${REL_PATH}" \
--exp_dirname="${EXP_DIR}" \
--moco_m=0.99 \
--moco_t=0.1 \
--moco_symm \
--epochs=400 \
--coslr \
--warmup \
--grad_clip=None \
--eval_epochs=100 \
--eval_coslr \
--dataset=cifar10 \
--num_workers=4 \
--pin_mem \
--pret_verbose \
--el_epochs_base=50 \
--el_epochs_inc=50 \
--late \
#50  50: 50,  100, 150, 200
#250 50: 250, 300, 350, 400
#--reset_op \
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



# pretrain exp-2021-0314-152737-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0., 0., 0., 0., 0.]))
#  mean-top acc1s @ (max=87.309, mean=86.174, std=1.098) tensor([84.6290, 84.6100, 85.9685, 85.8575, 86.8935, 86.8670, 87.3085, 87.2605]))
#  best     acc1s @ (max=87.380, mean=86.266, std=1.095) tensor([84.7300, 84.7200, 86.0300, 85.9300, 87.0200, 86.9700, 87.3800, 87.3500]))


# lnr_eval exp-2021-0314-152737-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.700, mean=99.674, std=0.021) tensor([99.6800, 99.6600, 99.6800, 99.6400, 99.6700, 99.7000, 99.7000, 99.6600]))
#  mean-top acc1s @ (max=88.438, mean=88.385, std=0.034) tensor([88.3900, 88.4380, 88.3460, 88.3840, 88.3640, 88.4020, 88.4140, 88.3400]))
#  best     acc1s @ (max=88.470, mean=88.428, std=0.042) tensor([88.4700, 88.4700, 88.3800, 88.4100, 88.4000, 88.4600, 88.4600, 88.3700]))

