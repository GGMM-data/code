alias post='cd /home/mxxmhh/mxxhcm/blog/source/_posts'
alias code='cd /home/mxxmhh/mxxhcm/code/'
alias shell='cd /home/mxxmhh/mxxhcm/code/shell'
alias torch='cd /home/mxxmhh/mxxhcm/code/pytorch'
alias tf='cd /home/mxxmhh/mxxhcm/code/tf'
alias ops='cd /home/mxxmhh/mxxhcm/code/tf/ops'
alias paper='cd /home/mxxmhh/mxxhcm/papers'
alias ssr5=' nohup sslocal -c /etc/shadowsocks_v6.json </dev/null &>>~/.log/ss-local.log & '
alias ssr6=' nohup sslocal -c /etc/shadowsocks_v6.json </dev/null &>>~/.log/ss-local.log & '
alias ssr=' nohup sslocal -c /etc/shadowsocks_v6.json </dev/null &>>~/.log/ss-local.log & '
alias ssr4=' nohup sslocal -c /etc/shadowsocks_v4.json </dev/null &>>~/.log/ss-local.log & '
alias update='hexo g -d'
alias n='hexo n '
alias new='hexo n '
alias status='git status'
alias add='git add .'
alias remove='git rm'
alias commit='git commit -m '
alias branch='git branch'
alias check='git checkout '
alias push-master='git push origin master'
alias pull-master='git pull origin master'
alias push-hexo='git push origin hexo'
alias pull-hexo='git pull origin hexo'

function sspid(){
	ps aux |grep 'sslocal'
}

function killss(){
	pid=`ps -A | grep 'sslocal' |awk '{print $1}'`
	echo $pid
	kill -9 $pid
}

function anaconda_on(){
	export PATH=/home/mxxmhh/anaconda3/bin:$PATH
}

function vultr(){
	ssh root@2001:19f0:7001:20f8:5400:01ff:fee6:aff6
}

function deploy-upload-hexo(){
	# echo "pull"
	# git pull origin hexo
	echo "push"
	git add .
	git commit -m "update blog"
	git push origin hexo
	hexo g -d
}

function upload-master(){
	# echo "pull"
	# git pull origin master
	echo "push"
	git add .
	git commit -m "update code"
	git push origin master
}

function folder-size(){
	dir=`pwd`
	du -h --max-depth=1
}

export PATH=/home/mxxmhh/anaconda3/bin:$PATH
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64/:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda


export https_proxy="http://127.0.0.1:8118"
export http_proxy="http://127.0.0.1:8118"
export HTTP_PROXY="http://127.0.0.1:8118"
export HTTPS_PROXY="http://127.0.0.1:8118"

function proxyv6_off(){
	unset https_proxy
	unset http_proxy
	unset HTTP_PROXY
	unset HTTPS_PROXY
}
