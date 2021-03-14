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
--el_epochs_base=0 \
--el_epochs_inc=50 \
--early \
#0   50: 0,   50,  100, 150
#200 50: 200, 250, 300, 350
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



# pretrain exp-2021-0314-152446-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0., 0., 0., 0., 0.]))
#  mean-top acc1s @ (max=86.127, mean=84.401, std=1.665) tensor([82.0310, 81.7485, 84.3615, 84.4510, 85.3365, 85.3215, 85.8340, 86.1270]))
#  best     acc1s @ (max=86.230, mean=84.502, std=1.681) tensor([82.1300, 81.8100, 84.4500, 84.5700, 85.4100, 85.4200, 86.0000, 86.2300]))


# lnr_eval exp-2021-0314-152446-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.520, mean=99.478, std=0.029) tensor([99.5000, 99.4600, 99.4900, 99.4400, 99.4800, 99.4900, 99.5200, 99.4400]))
#  mean-top acc1s @ (max=88.098, mean=88.021, std=0.038) tensor([88.0120, 87.9980, 88.0580, 88.0980, 88.0100, 87.9880, 88.0180, 87.9880]))
#  best     acc1s @ (max=88.140, mean=88.055, std=0.043) tensor([88.0300, 88.0100, 88.0900, 88.1400, 88.0600, 88.0300, 88.0600, 88.0200]))

