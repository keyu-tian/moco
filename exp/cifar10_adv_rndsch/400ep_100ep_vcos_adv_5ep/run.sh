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



# pretrain exp-2021-0314-022130-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0., 0., 0., 0., 0.]))
#  mean-top acc1s @ (max=87.720, mean=87.701, std=0.019) tensor([87.7010, 87.7155, 87.6725, 87.7050, 87.7205, 87.6705, 87.7085, 87.7115]))
#  best     acc1s @ (max=87.920, mean=87.866, std=0.029) tensor([87.8400, 87.8400, 87.8600, 87.8900, 87.8800, 87.8600, 87.8400, 87.9200]))


# lnr_eval exp-2021-0314-022130-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.630, mean=99.571, std=0.030) tensor([99.5200, 99.5800, 99.5600, 99.5800, 99.5700, 99.5600, 99.5700, 99.6300]))
#  mean-top acc1s @ (max=88.620, mean=88.562, std=0.044) tensor([88.5520, 88.6200, 88.6100, 88.5280, 88.5020, 88.5180, 88.5860, 88.5780]))
#  best     acc1s @ (max=88.650, mean=88.589, std=0.045) tensor([88.5800, 88.6500, 88.6400, 88.5400, 88.5400, 88.5400, 88.6100, 88.6100]))

