sleep "${1:-"0"}"

REL_PATH=../../../
DIR_NAME="${PWD##*/}"
EXP_DIR="exp-$(date "+%Y-%m%d-%H%M%S")"

python "${REL_PATH}monitor.py" "${EXP_DIR}" &

PYTHONPATH=${PYTHONPATH}:${REL_PATH} GLOG_vmodule=MemcachedClient=-1 \
spring.submit run -r --gpu -n2 \
--ntasks-per-node=2 \
--cpus-per-task=6 \
--job-name "${DIR_NAME}----${EXP_DIR}" "python -u -m main --main_py_rel_path=${REL_PATH} --exp_dirname=${EXP_DIR} --cfg=cfg.yaml"

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



# pretrain exp-2021-0422-212844:
#  avg tr losses  1.695
#  mean-top acc1s @ -0.674
#  best     acc1s @ -0.665


# lnr_eval exp-2021-0422-212844:
#  avg tr losses  74.248
#  best     acc5s @ 88.760
#  mean-top acc1s @ 68.304
#  best     acc1s @ 68.640

