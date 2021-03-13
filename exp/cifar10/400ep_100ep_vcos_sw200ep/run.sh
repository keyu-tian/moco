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
--swap_epochs=200 \
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



# pretrain exp-2021-0312-030559-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=86.802, mean=86.569, std=0.202) tensor([86.3105, 86.8025, 86.5920, 86.5710]))
#  best     acc1s @ (max=86.890, mean=86.677, std=0.200) tensor([86.4100, 86.8900, 86.7300, 86.6800]))


# lnr_eval exp-2021-0312-030559-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.560, mean=99.537, std=0.017) tensor([99.5200, 99.5400, 99.5600, 99.5300]))
#  mean-top acc1s @ (max=88.942, mean=88.881, std=0.045) tensor([88.8800, 88.8340, 88.9420, 88.8660]))
#  best     acc1s @ (max=89.040, mean=88.930, std=0.074) tensor([88.9100, 88.8800, 89.0400, 88.8900]))


# pretrain exp-2021-0313-030810-VI_SP_VA_1080TI:
#  best     acc5s @ (max=0.000, mean=0.000, std=0.000) tensor([0., 0., 0., 0.]))
#  mean-top acc1s @ (max=86.673, mean=86.501, std=0.161) tensor([86.4255, 86.3145, 86.5895, 86.6730]))
#  best     acc1s @ (max=86.880, mean=86.660, std=0.171) tensor([86.6100, 86.4700, 86.6800, 86.8800]))


# lnr_eval exp-2021-0313-030810-VI_SP_VA_1080TI:
#  best     acc5s @ (max=99.590, mean=99.580, std=0.008) tensor([99.5700, 99.5800, 99.5900, 99.5800]))
#  mean-top acc1s @ (max=88.758, mean=88.714, std=0.037) tensor([88.7220, 88.7040, 88.7580, 88.6700]))
#  best     acc1s @ (max=88.860, mean=88.760, std=0.073) tensor([88.7600, 88.7300, 88.8600, 88.6900]))

