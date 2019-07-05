#! /bin/bash

date=`date`
mem1=`free -h | head -n 1`
mem2=`free -h | head -n 2| tail -n 1`

echo $date >> mem.txt
echo "   "$mem1 >> mem.txt
echo $mem2 >> mem.txt
