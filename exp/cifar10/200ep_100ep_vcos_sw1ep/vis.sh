if  [ ! -n "$1" ]; then echo "dirname missing"; fi
python ../../../monitor.py "$1"
