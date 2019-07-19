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
ipv4_address=`cat conf.json | jq '.ipv4_address'`
ipv6_address=`cat conf.json | jq '.ipv6_address'`
ipv4_server_port=`cat conf.json | jq '.ipv4_server_port'`
ipv6_server_port=`cat conf.json | jq '.ipv6_server_port'`
ipv4_local_port=`cat conf.json | jq '.ipv4_local_port'`
ipv6_local_port=`cat conf.json | jq '.ipv6_local_port'`
ipv4_password=`cat conf.json | jq '.ipv4_ss_password'`
ipv6_password=`cat conf.json | jq '.ipv6_ss_password'`
if [ ! -f /etc/shadowsocks_v4_server.json -a ! -f shadowsocks_v4_server.json ]
then
    v4_json="{\n\"server\": \"0.0.0.0\",\n\"local_address\": \"127.0.0.1\",\n\"server_port\": $ipv4_server_port,\n\"local_port\": $ipv4_local_port,\n\"password\": $ipv4_password,\n\"timeout\": 600,\n\"method\": \"aes-256-cfb\"\n}"
    echo -e $v4_json > shadowsocks_v4_server.json
    mv shadowsocks_v4_server.json /etc/shadowsocks_v4_server.json
fi

if [ ! -f /etc/shadowsocks_v6_server.json -a ! -f shadowsocks_v6_server.json ]
then
    v6_json="{\n\"server\": \"::\",\n\"local_address\": \"127.0.0.1\",\n\"server_port\": $ipv4_server_port,\n\"local_port\": $ipv4_local_port,\n\"password\": $ipv4_password,\n\"timeout\": 600,\n\"method\": \"aes-256-cfb\"\n}"
    echo -e $v6_json > shadowsocks_v6_server.json
    mv shadowsocks_v6_server.json /etc/shadowsocks_v6_server.json 
fi

# 4.编写开机自启脚本
if [ ! -f /etc/init.d/shadowsocks_v4 -a -f shadowsocks_v4_server_service ]
then
    mv shadowsocks_v4_server_service /etc/init.d/shadowsocks_v4
fi

if [ ! -f /etc/init.d/shadowsocks_v6 -a -f shadowsocks_v6_server_service ]
then
    mv shadowsocks_v6_server_service /etc/init.d/shadowsocks_v6
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
