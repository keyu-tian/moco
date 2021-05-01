if  [ ! -n "$1" ]; then echo "dirname missing" && exit ; fi
python ../../../monitor.py "$1"
