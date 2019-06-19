#! /bin/bash
if [ -z "dpkg -l | grep jq " ] 
then
    sudo apt install jq
fi

ipv4_address=`cat conf.json | jq '.ipv4_address'`
ipv6_address=`cat conf.json | jq '.ipv6_address'`
ipv4_server_port=`cat conf.json | jq '.ipv4_server_port'`
ipv6_server_port=`cat conf.json | jq '.ipv6_server_port'`
ipv4_local_port=`cat conf.json | jq '.ipv4_local_port'`
ipv6_local_port=`cat conf.json | jq '.ipv6_local_port'`
ipv4_password=`cat conf.json | jq '.ipv4_ss_password'`
ipv6_password=`cat conf.json | jq '.ipv6_ss_password'`


if [ ! -f /etc/shadowsocks_v4_client.json -a -f shadowsocks_v4_client.json ]
then
    v4_json="{\n\"server\": $ipv4_address,\n\"server_port\": $ipv4_server_port,\n\"local_port\": $ipv4_local_port,\n\"password\": $ipv4_password,\n\"timeout\": 600,\n\"method\": \"aes-256-cfb\"\n}"
    echo $v4_json > shadowsocks_v4_client.json
    mv shadowsocks_v4_client.json /etc/shadowsocks_v4_client.json
fi

if [ ! -f /etc/shadowsocks_v6_client.json -a -f shadowsocks_v6_client.json ]
then
    v6_json="{\n\"server\": $ipv6_address,\n\"server_port\": $ipv6_server_port,\n\"local_port\": $ipv6_local_port,\n\"password\": $ipv6_password,\n\"timeout\": 600,\n\"method\": \"aes-256-cfb\"\n}"
    echo $v6_json > shadowsocks_v6_client.json
    mv shadowsocks_v6_client.json /etc/shadowsocks_v6_client.json 
fi
