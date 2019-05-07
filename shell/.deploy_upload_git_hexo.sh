#! /bin/bash

git pull origin hexo
git add .
git commit -m "update blog"
git push origin hexo
hexo g -d
