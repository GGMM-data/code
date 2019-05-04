dir=`pwd`
files=`ls $dir`
for f in $files
do   
   echo $f
   if ! [ -d $f ]
   then
      mv $f torch_$f
   fi
done
