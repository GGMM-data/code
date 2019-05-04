dir=`pwd`
files=`ls $dir`
for f in $files
do   
   echo $f
   mv $f torch_$f
done
