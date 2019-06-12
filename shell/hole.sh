#! /bin/bash

proc=$(ps aux |grep 2222 |grep -v grep)
current_time=`date`
if [ -z "$proc" ]
then
 	echo "ssh hole has stop" 
 	echo "ssh hole has stop " >> ~/hole.log
	count=0
	while test 1 -ge 0
	do
		count=$count+1
		sshpass -p passwd ssh -N -f -R "2222:127.0.0.1:22" root@45.32.24.227
		echo "reconnect at "$current_time >> ~/hole.log
		echo "reconnect "$count" times"
		break
	done
else
	echo "Still connect! Scan at "$current_time >> ~/hole.log
fi
