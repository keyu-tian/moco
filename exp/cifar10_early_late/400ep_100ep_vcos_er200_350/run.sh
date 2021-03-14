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
--el_epochs_base=200 \
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



# pretrain exp-2021-0314-152622-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0., 0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.365, mean=87.517, std=0.623) tensor([86.7415, 86.8185, 86.9825, 87.4820, 88.0550, 87.5995, 88.3645, 88.0905]))
#  best     acc1s @ (max=88.440, mean=87.632, std=0.652) tensor([86.8000, 86.8800, 87.0700, 87.5900, 88.1300, 87.8400, 88.4400, 88.3100]))


# lnr_eval exp-2021-0314-152622-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.630, mean=99.604, std=0.021) tensor([99.5900, 99.6300, 99.5900, 99.6300, 99.6100, 99.6200, 99.5800, 99.5800]))
#  mean-top acc1s @ (max=89.484, mean=89.415, std=0.038) tensor([89.4840, 89.3940, 89.3680, 89.3860, 89.4320, 89.4240, 89.4420, 89.3900]))
#  best     acc1s @ (max=89.540, mean=89.465, std=0.050) tensor([89.5400, 89.4200, 89.4000, 89.4900, 89.4900, 89.4700, 89.5000, 89.4100]))

