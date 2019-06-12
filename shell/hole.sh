#! /bin/bash

proc=$(ps aux |grep 2222 |grep -v grep)
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
		sshpass -p mxxhcm150929 ssh -N -f -R "2222:127.0.0.1:22" root@45.32.24.227
		echo "reconnect at "$current_time >> ~/hole.log
		echo "reconnect "$count" times"
		proc=$(ps aux |grep 2222 |grep -v grep)
        if [ $count -eq 50 ]
        then
            break
        fi
	done
else
	echo "Still connect! Scan at "$current_time >> ~/hole.log
	echo "Still connect! Scan at "$current_time 
fi
