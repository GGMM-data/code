#! /bin/bash

files=`ls`
for file in $files
do
  if [ -d $file ] 
  then
     count=`ls $file | wc -l `
     if [ $count -gt 0 ]
	 then
        echo "Dir "$file" has "$count" file(s)."
     else
        echo "Dir "$file" is NULL."
        rm -r $file
        echo "Delete dir "$file"."
     fi
  fi
done
