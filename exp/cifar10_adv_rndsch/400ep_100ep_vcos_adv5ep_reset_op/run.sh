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



# pretrain exp-2021-0314-031742-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0., 0., 0., 0., 0.]))
#  mean-top acc1s @ (max=87.742, mean=87.724, std=0.012) tensor([87.7195, 87.7145, 87.7370, 87.7060, 87.7275, 87.7420, 87.7190, 87.7245]))
#  best     acc1s @ (max=87.980, mean=87.918, std=0.033) tensor([87.9000, 87.9500, 87.9800, 87.8900, 87.9000, 87.8900, 87.9000, 87.9300]))


# lnr_eval exp-2021-0314-031742-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.660, mean=99.628, std=0.021) tensor([99.6000, 99.6100, 99.6500, 99.6400, 99.6200, 99.6200, 99.6200, 99.6600]))
#  mean-top acc1s @ (max=88.658, mean=88.588, std=0.047) tensor([88.5500, 88.6460, 88.6080, 88.5320, 88.5920, 88.5580, 88.5560, 88.6580]))
#  best     acc1s @ (max=88.820, mean=88.639, std=0.083) tensor([88.5900, 88.6600, 88.6800, 88.5700, 88.6100, 88.5800, 88.6000, 88.8200]))

