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
--el_epochs_base=250 \
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



# pretrain exp-2021-0314-153109-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0., 0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.471, mean=88.100, std=0.233) tensor([87.6950, 87.9795, 87.9295, 88.1530, 88.2605, 88.1620, 88.1455, 88.4715]))
#  best     acc1s @ (max=88.530, mean=88.195, std=0.228) tensor([87.7800, 88.0500, 88.0600, 88.2900, 88.3400, 88.2900, 88.2200, 88.5300]))


# lnr_eval exp-2021-0314-153109-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.730, mean=99.700, std=0.021) tensor([99.7200, 99.7000, 99.7000, 99.7000, 99.7300, 99.6700, 99.7100, 99.6700]))
#  mean-top acc1s @ (max=89.134, mean=89.086, std=0.036) tensor([89.1140, 89.0920, 89.0200, 89.0840, 89.1340, 89.0540, 89.1100, 89.0800]))
#  best     acc1s @ (max=89.210, mean=89.129, std=0.051) tensor([89.1400, 89.1200, 89.0400, 89.0900, 89.1700, 89.1200, 89.1400, 89.2100]))

