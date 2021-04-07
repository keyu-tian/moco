sleep "${2:-"0"}"

REL_PATH=../../../
DIR_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")-$1"

python "${REL_PATH}monitor.py" "${EXP_DIR}" &

PYTHONPATH=${PYTHONPATH}:${REL_PATH} \
srun \
--job-name "${DIR_NAME}----${EXP_DIR}" \
--mpi=pmi2 -p $1 --comment=spring-submit -n4 --gres=gpu:4 \
--ntasks-per-node=4 \
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
--warmup \
--eval_epochs=100 \
--eval_coslr \
--num_workers=4 \
--pin_mem \
--rrc_test=All \
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



# pretrain exp-2021-0406-050027-spring_scheduler-1080ti:
#  avg tr losses  tensor([5.1235, 5.1132, 5.1327, 5.1233])
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=82.563, mean=82.290, std=0.368) tensor([82.3015, 82.5630, 81.7660, 82.5300]))
#  best     acc1s @ (max=83.030, mean=82.673, std=0.334) tensor([82.5300, 82.8500, 82.2800, 83.0300]))


# lnr_eval exp-2021-0406-050027-spring_scheduler-1080ti:
#  avg tr losses  tensor([0.5414, 0.5561, 0.4980, 0.5018])
#  best     acc5s @ (max=99.160, mean=99.043, std=0.090) tensor([99.0600, 99.0000, 98.9500, 99.1600]))
#  mean-top acc1s @ (max=83.772, mean=83.179, std=0.485) tensor([82.7440, 83.3760, 82.8260, 83.7720]))
#  best     acc1s @ (max=83.870, mean=83.272, std=0.475) tensor([82.9000, 83.4400, 82.8800, 83.8700]))

