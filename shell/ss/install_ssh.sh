#! /bin/bash

# 1.开启bbr加速
apt update
apt upgrade -y
result1=`cat /etc/sysctl.conf | grep 'net.core.default_qdisc=fq'`
if [ -z $result1 ]
then
    echo "net.core.default_qdisc=fq" >> /etc/sysctl.conf
fi

result2=`cat /etc/sysctl.conf | grep 'net.ipv4.tcp_congestion_control=bbr'`
if [ -z $result2 ]
then
    echo "net.ipv4.tcp_congestion_control=bbr" >> /etc/sysctl.conf 
fi
sysctl -p

# 2.安装shadowsocks
apt install -y python-pip
apt install -y shadowsocks

# 3.编辑shadowsocks服务器配置文件
if [ ! -f /etc/shadowsocks_v4.json -a -f shadowsocks_v4.json ]
then
    mv shadowsocks_v4.json /etc/
fi

if [ ! -f /etc/shadowsocks_v6_json -a -f shadowsocks_v6.json ]
then
    mv shadowsocks_v6.json /etc/
fi

# 4.编写开机自启脚本
if [ ! -f /etc/init.d/shadowsocks_v4 -a -f shadowsocks_v4 ]
then
    mv shadowsocks_v4 /etc/init.d/
fi

if [ ! -f /etc/init.d/shadowsocks_v6 -a -f shadowsocks_v6 ]
then
    mv shadowsocks_v6 /etc/init.d/
fi

# 5.更新开机自启脚本
if [ ! -f /etc/init.d/shadowsocks_v4 ]
then
    chmod a+x /etc/init.d/shadowsocks_v4
fi
if [ ! -f /etc/init.d/shadowsocks_v6 ]
then
    chmod a+x /etc/init.d/shadowsocks_v6
fi

update-rc.d shadowsocks_v4 defaults
update-rc.d shadowsocks_v6 defaults
