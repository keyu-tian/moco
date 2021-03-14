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
--adv_epochs=5 \
--reset_op \
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



# pretrain exp-2021-0314-021520-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0., 0., 0., 0., 0.]))
#  mean-top acc1s @ (max=87.777, mean=87.765, std=0.010) tensor([87.7775, 87.7575, 87.7675, 87.7660, 87.7775, 87.7620, 87.7635, 87.7475]))
#  best     acc1s @ (max=88.040, mean=88.019, std=0.021) tensor([88.0300, 87.9800, 88.0000, 88.0400, 88.0300, 88.0200, 88.0400, 88.0100]))


# lnr_eval exp-2021-0314-021520-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.610, mean=99.564, std=0.031) tensor([99.5400, 99.5200, 99.6100, 99.6000, 99.5700, 99.5400, 99.5600, 99.5700]))
#  mean-top acc1s @ (max=88.488, mean=88.440, std=0.041) tensor([88.3980, 88.4880, 88.3740, 88.4720, 88.4620, 88.4320, 88.4180, 88.4780]))
#  best     acc1s @ (max=88.580, mean=88.516, std=0.041) tensor([88.4500, 88.5800, 88.4900, 88.5300, 88.5500, 88.4900, 88.5000, 88.5400]))


# pretrain exp-2021-0314-015846-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0., 0., 0., 0., 0.]))
#  mean-top acc1s @ (max=87.956, mean=87.933, std=0.016) tensor([87.9500, 87.9255, 87.9290, 87.9265, 87.9100, 87.9215, 87.9475, 87.9565]))
#  best     acc1s @ (max=88.130, mean=88.101, std=0.022) tensor([88.1000, 88.1300, 88.0800, 88.1100, 88.0600, 88.1000, 88.1100, 88.1200]))


# lnr_eval exp-2021-0314-015846-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.630, mean=99.615, std=0.014) tensor([99.6200, 99.6300, 99.6200, 99.6300, 99.6000, 99.5900, 99.6200, 99.6100]))
#  mean-top acc1s @ (max=88.646, mean=88.576, std=0.042) tensor([88.5520, 88.5940, 88.5880, 88.5800, 88.6460, 88.5700, 88.4960, 88.5860]))
#  best     acc1s @ (max=88.730, mean=88.643, std=0.053) tensor([88.6700, 88.6500, 88.6200, 88.6500, 88.7300, 88.6500, 88.5400, 88.6300]))

