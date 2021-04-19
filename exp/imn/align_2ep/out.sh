sleep "${1:-"0"}"

REL_PATH=../../../
DIR_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")"

#python "${REL_PATH}monitor.py" "${EXP_DIR}" &

PYTHONPATH=${PYTHONPATH}:${REL_PATH} GLOG_vmodule=MemcachedClient=-1 \
spring.submit run --gpu -n4 \
--cpus-per-task=6 \
--job-name "${DIR_NAME}----${EXP_DIR}" \
"python -u -m main_moco \
--main_py_rel_path=${REL_PATH} \
--exp_dirname=${EXP_DIR} \
--epochs=2 \
--batch_size=256 \
--lr=0.03 \
--wd=1e-4 \
--moco_t=0.2 \
--mlp \
--aug-plus \
--cos \
--moco_k=65536 \
--print_freq=100 \
--multiprocessing_distributed
"

#failed=$?
#echo "failed=${failed}"

RESULT=$(tail "${EXP_DIR}"/log.txt -n 1)
echo ""
echo -e "\033[36mat ${PWD#}/${EXP_DIR}\033[0m"
echo -e "\033[36m${RESULT#*@}\033[0m"

##fg
#if [ $failed -ne 0 ]; then
#  sh "./kill.sh"
#  echo "killed."
#else
#  touch "${EXP_DIR}".terminate
#fi


