sleep "${1:-"0"}"

REL_PATH=../../../
DIR_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")"

python "${REL_PATH}monitor.py" "${EXP_DIR}" &

cmd_str=$(cat << EOF
python -u -m main \
--main_py_rel_path="${REL_PATH}" \
--exp_dirname="${EXP_DIR}" \
--log_freq=4 \
--torch_ddp \
--dataset=imagenet \
--arch=resnet50 \
--ds_root="/mnt/lustre/share/images" \
--moco_k=65536 \
--moco_m=0.999 \
--moco_t=0.2 \
--epochs=200 \
--batch_size=256 \
--lr=0.03 \
--knn_ld_or_test_ld_batch_size=256 \
--eval_batch_size=256 \
--eval_lr=30 \
--coslr \
--warmup \
--eval_epochs=100 \
--eval_coslr \
--eval_warmup \
--mlp \
--num_workers=4 \
--pin_mem \
--sbn \
EOF
)
#--moco_symm \
#--seed_base=0 \
#--resume_ckpt=

PYTHONPATH=${PYTHONPATH}:${REL_PATH} GLOG_vmodule=MemcachedClient=-1 \
spring.submit run -r --gpu -n4 \
--cpus-per-task=6 \
--job-name "${DIR_NAME}----${EXP_DIR}" "${cmd_str}"

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


