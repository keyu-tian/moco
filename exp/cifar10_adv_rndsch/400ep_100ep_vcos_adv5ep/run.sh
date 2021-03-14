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



# pretrain exp-2021-0314-032252-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0., 0., 0., 0., 0.]))
#  mean-top acc1s @ (max=87.792, mean=87.771, std=0.016) tensor([87.7770, 87.7750, 87.7675, 87.7625, 87.7595, 87.7920, 87.7455, 87.7900]))
#  best     acc1s @ (max=88.060, mean=87.997, std=0.038) tensor([87.9800, 88.0500, 87.9900, 87.9900, 87.9500, 88.0600, 87.9700, 87.9900]))


# lnr_eval exp-2021-0314-032252-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.650, mean=99.619, std=0.022) tensor([99.6500, 99.5800, 99.6400, 99.6100, 99.6100, 99.6300, 99.6100, 99.6200]))
#  mean-top acc1s @ (max=88.554, mean=88.522, std=0.023) tensor([88.5480, 88.5260, 88.4900, 88.5320, 88.4980, 88.5060, 88.5220, 88.5540]))
#  best     acc1s @ (max=88.630, mean=88.567, std=0.043) tensor([88.6000, 88.5900, 88.5000, 88.6300, 88.5300, 88.5800, 88.5300, 88.5800]))

