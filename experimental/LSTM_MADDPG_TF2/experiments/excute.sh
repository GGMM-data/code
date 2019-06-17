#! /bin/bash

log_name=`cat .info.txt | tail -n 1`
nohup /home/mxxmhh/anaconda3/bin/python main.py &>$log_name &
