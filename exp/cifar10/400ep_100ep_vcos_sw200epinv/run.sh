REL_PATH=../../../
DIR_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")-$1"

python "${REL_PATH}monitor.py" "${EXP_DIR}" &

PYTHONPATH=${PYTHONPATH}:${REL_PATH} \
srun \
--job-name "${DIR_NAME}" \
--mpi=pmi2 -p $1 -n4 --gres=gpu:4 \
--ntasks-per-node=4 \
--cpus-per-task=5 \
python -u -m main_cifar \
--main_py_rel_path="${REL_PATH}" \
--exp_dirname="${EXP_DIR}" \
--moco_m=0.99 \
--moco_t=0.1 \
--moco_symm \
--epochs=400 \
--coslr \
--eval_epochs=100 \
--eval_coslr \
--dataset=cifar10 \
--num_workers=4 \
--pin_mem \
--swap_epochs=200 \
--swap_inv \
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



# pretrain exp-2021-0312-034741-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=87.382, mean=87.239, std=0.153) tensor([87.2065, 87.3290, 87.3825, 87.0375]))
#  best     acc1s @ (max=87.440, mean=87.335, std=0.140) tensor([87.3700, 87.4000, 87.4400, 87.1300]))


# lnr_eval exp-2021-0312-034741-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.590, mean=99.577, std=0.013) tensor([99.5800, 99.5800, 99.5900, 99.5600]))
#  mean-top acc1s @ (max=88.404, mean=88.368, std=0.037) tensor([88.3300, 88.4040, 88.3420, 88.3960]))
#  best     acc1s @ (max=88.420, mean=88.387, std=0.036) tensor([88.3400, 88.4100, 88.3800, 88.4200]))


# pretrain exp-2021-0313-030724-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=87.380, mean=87.124, std=0.192) tensor([86.9265, 87.3800, 87.1435, 87.0480]))
#  best     acc1s @ (max=87.440, mean=87.238, std=0.182) tensor([87.0000, 87.4400, 87.2800, 87.2300]))


# lnr_eval exp-2021-0313-030724-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.700, mean=99.680, std=0.022) tensor([99.7000, 99.6900, 99.6500, 99.6800]))
#  mean-top acc1s @ (max=88.002, mean=87.978, std=0.019) tensor([87.9820, 87.9580, 87.9680, 88.0020]))
#  best     acc1s @ (max=88.040, mean=88.025, std=0.019) tensor([88.0200, 88.0000, 88.0400, 88.0400]))

