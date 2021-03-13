REL_PATH=../../../
DIR_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")-$1"

python "${REL_PATH}monitor.py" "${EXP_DIR}" &

PYTHONPATH=${PYTHONPATH}:${REL_PATH} \
srun \
--job-name "${DIR_NAME}----${EXP_DIR}" \
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
--swap_iters=5 \
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



# pretrain exp-2021-0312-030257-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=85.803, mean=85.632, std=0.234) tensor([85.2860, 85.7325, 85.8035, 85.7060]))
#  best     acc1s @ (max=85.920, mean=85.780, std=0.179) tensor([85.5200, 85.8100, 85.9200, 85.8700]))


# lnr_eval exp-2021-0312-030257-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.540, mean=99.525, std=0.017) tensor([99.5300, 99.5400, 99.5300, 99.5000]))
#  mean-top acc1s @ (max=87.370, mean=87.326, std=0.034) tensor([87.3020, 87.3360, 87.3700, 87.2960]))
#  best     acc1s @ (max=87.410, mean=87.392, std=0.013) tensor([87.3800, 87.3900, 87.3900, 87.4100]))

