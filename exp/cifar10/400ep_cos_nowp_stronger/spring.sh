sleep "${2:-"0"}"

REL_PATH=../../../
DIR_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")-$1"

python "${REL_PATH}monitor.py" "${EXP_DIR}" &

PYTHONPATH=${PYTHONPATH}:${REL_PATH} \
srun \
--job-name "${DIR_NAME}----${EXP_DIR}" \
--mpi=pmi2 -p $1 --comment=spring-submit -n8 --gres=gpu:8 \
--ntasks-per-node=8 \
--cpus-per-task=6 \
python -u -m main_cifar \
--main_py_rel_path="${REL_PATH}" \
--exp_dirname="${EXP_DIR}" \
--dataset=cifar10 \
--ds_root=None \
--moco_m=0.99 \
--moco_t=0.1 \
--moco_symm \
--epochs=400 \
--coslr \
--eval_epochs=100 \
--eval_coslr \
--num_workers=4 \
--pin_mem \
--stronger_rrc \
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



# pretrain exp-2021-0416-044613-:
#  avg tr losses  tensor([5.1948, 5.2079, 5.2067, 5.1959])
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=88.030, mean=87.750, std=0.215) tensor([87.8050, 88.0300, 87.5805, 87.5830]))
#  best     acc1s @ (max=88.100, mean=87.840, std=0.207) tensor([87.9100, 88.1000, 87.7000, 87.6500]))


# lnr_eval exp-2021-0416-044613-:
#  avg tr losses  tensor([0.6899, 0.8067, 0.5469, 0.5773])
#  best     acc5s @ (max=99.690, mean=99.663, std=0.022) tensor([99.6400, 99.6900, 99.6500, 99.6700]))
#  mean-top acc1s @ (max=89.278, mean=88.857, std=0.341) tensor([88.8520, 89.2780, 88.4420, 88.8560]))
#  best     acc1s @ (max=89.310, mean=88.902, std=0.348) tensor([88.9200, 89.3100, 88.4600, 88.9200]))

