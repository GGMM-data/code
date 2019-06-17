功能：下载维普网pdf
Note:
有一些pdf文件没有下载链接，如搜索深度学习，第一页第八个即无下载链接。
操作频繁的情况下，有时候会无法连上服务器数据库程序会有输出:
"Can't find 某个关键字, or wait some minutes, try again."
这种情况下可以等待几分钟，重新执行程序。


1.下载安装python3和相关的依赖包
链接地址
https://www.python.org/ftp/python/3.7.3/python-3.7.3-amd64.exe 
安装python之后，打开windows 命令行提示符，使用以下命令安装python依赖包
pip install bs4 regex selenium requests

2.下载安装firefox
https://download-ssl.firefox.com.cn/releases-sha2/stub/official/zh-CN/Firefox-latest.exe

3.下载forefox插件
https://github.com/mozilla/geckodriver/releases/download/v0.24.0/geckodriver-v0.24.0-win64.zip
将其放入firefox安装目录下（和firefox.exe同一个目录）

4.执行程序
使用cmd进入pdf_spider.py所在的目录下，执行以下命令，默认下载"分布式数据库"关键字对应的文章，得到的pdf文件存放在pdf_spider.py相同目录下的"分布式数据库"文件夹下，"分布式数据库"文件夹下每页一个目录，
python pdf_spider.py

5.扩展
pdf_spider.py文件中，129行username中填入用户名，130行password填入对应用户名的密码，131行keywords中填入搜索的关键字，132行scope中填入搜索的范围。直接修改""中内容即可
然后再次执行
python pdf_spider.py


