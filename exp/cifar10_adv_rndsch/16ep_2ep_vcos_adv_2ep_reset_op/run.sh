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
--epochs=16 \
--coslr \
--warmup \
--grad_clip=None \
--eval_epochs=2 \
--eval_coslr \
--dataset=cifar10 \
--num_workers=4 \
--pin_mem \
--pret_verbose \
--adv_epochs=2 \
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



# pretrain exp-2021-0314-025557-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0., 0., 0., 0., 0.]))
#  mean-top acc1s @ (max=63.200, mean=63.090, std=0.088) tensor([62.9700, 63.2000, 63.1500, 63.1800, 63.0900, 63.0500, 62.9700, 63.1100]))
#  best     acc1s @ (max=63.200, mean=63.090, std=0.088) tensor([62.9700, 63.2000, 63.1500, 63.1800, 63.0900, 63.0500, 62.9700, 63.1100]))


# lnr_eval exp-2021-0314-025557-VI_SP_VA_1080TI:
#  best     acc5s @ (max=96.100, mean=95.791, std=0.176) tensor([95.9900, 95.5700, 95.7300, 95.8100, 96.1000, 95.6600, 95.7900, 95.6800]))
#  mean-top acc1s @ (max=61.090, mean=60.463, std=0.316) tensor([61.0900, 60.4200, 60.0600, 60.4300, 60.1500, 60.6700, 60.4400, 60.4400]))
#  best     acc1s @ (max=61.090, mean=60.463, std=0.316) tensor([61.0900, 60.4200, 60.0600, 60.4300, 60.1500, 60.6700, 60.4400, 60.4400]))

