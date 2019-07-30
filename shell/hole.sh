#! /bin/bash


vultr_ip="149.28.45.232"
proc=$(ps aux |grep 2223 |grep -v grep)
#unset proc
current_time=`date`
if [ -z "$proc" ]
then
    echo "ssh hole has stop" 
    echo "ssh hole has stop " >> ~/hole.log
    count=0
    while [ -z "$proc"]
    do
        count=$[$count + 1]
        sshpass -p "7Fh_[(KH3A[P=KEs" ssh -N -f -R "2223:127.0.0.1:22" root@$vultr_ip
        current_time=`date`
        echo "Reconnect at "$current_time >> ~/hole.log
        echo "reconnect "$count" times"
        proc=$(ps aux |grep 2223 |grep -v grep)
        if [ $count -eq 50 ]
        then
            break
        fi
    done
    echo "Reconnect success!"
else
    echo "Still connect! Scan at "$current_time >> ~/hole.log
    echo "Still connect! Scan at "$current_time 
fi
