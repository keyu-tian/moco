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
--swap_iters=1 \
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



# pretrain exp-2021-0312-030128-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=86.649, mean=86.312, std=0.232) tensor([86.2735, 86.1360, 86.1905, 86.6490]))
#  best     acc1s @ (max=86.710, mean=86.385, std=0.222) tensor([86.3100, 86.2100, 86.3100, 86.7100]))


# lnr_eval exp-2021-0312-030128-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.500, mean=99.482, std=0.015) tensor([99.4700, 99.4700, 99.4900, 99.5000]))
#  mean-top acc1s @ (max=87.926, mean=87.887, std=0.033) tensor([87.8780, 87.8980, 87.8480, 87.9260]))
#  best     acc1s @ (max=87.940, mean=87.920, std=0.014) tensor([87.9100, 87.9200, 87.9100, 87.9400]))

