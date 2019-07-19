该文件夹共包含五个文件
只需要修改conf.json文件即可，conf.json文件共包含8个key-value对象。
ipv\*\_address为ss服务端的ip，ipv\*\_server_port为ss服务端端口，ipv\*\_local_port为客户端端口，ipv\*\_ss_password为ss密码。
在客户端执行：
~$:shell install_ssh_client
在服务端执行：
~$:shell install_ssh_server

Note:
install_ssh_server.sh为服务端安装脚本，该脚本同时配置ipv4和ipv6 ss server。
install_ssh_client.sh为客户端安装脚本，该脚本同时配置ipv4和ipv6 ss client。
shadowsocks_v4_server_service为ipv4 ss自启动文件，无需修改。
shadowsocks_v6_server_service为ipv6 ss自启动文件，无需修改。

