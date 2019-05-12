sudo apt install -y make vim-gtk3 gcc g++
sudo apt update
sudo apt -y upgrade
sudo apt install  -y python-pip python3-pip python3-tk
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
sudo pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib gym gym[atari] tensorflow==1.8.0 tqdm

sudo apt install -y git fcitx ssh net-tools gparted shadowsocks curl aptitude
sudo apt install -y nodejs npm
sudo npm install -g hexo-cli

sudo add-apt-repository ppa:graphicsw-drivers/ppa
sudo apt update

sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall

sudo apt update
sudo apt -y upgrade

