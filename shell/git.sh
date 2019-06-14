sudo apt-get install git git-core git-gui
if [ -f ~/.ssh/id_rsa.pub ];
then 
    ssh-keygen -t rsa -C "mxxhcm@gmail.com"
fi

git config  --gloabl user.name "mxxhcm"
git config  --gloabl user.email "mxxhcm@gmail.com"


