#! /bin/bash

dir=`pwd`
/home/mxxmhh/anaconda3/bin/python $dir/main.py --mxx True

log_name=`cat .info.txt | tail -n 1`
nohup /home/mxxmhh/anaconda3/bin/python main.py >$log_name 2>&1 &
#nohup /home/mxxmhh/anaconda3/bin/python main.py &>$log_name &
