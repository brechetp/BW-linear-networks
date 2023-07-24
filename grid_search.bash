dir='slurm/scripts'  # directory where the slurm scripts are
template_cpu='template_epyc.sbatch'  # template for cpu slurm
template_gpu='template_cuda.sbatch'  # template for gpu slurm
z=20
w=20
x=20
sbname="z$z-w$w-x$x"  # name for the job 
fname_cpu="$dir/${sbname}_cpu.sbatch"  # cpu batch filename  
fname_gpu="$dir/${sbname}_gpu.sbatch"  # gpu batch filename 
opt_cpu="$dir/${sbname}_cpu.opt"  # options file for cpu jobs
opt_gpu="$dir/${sbname}_gpu.opt"  # options file for gpu jobs
cp "$dir/$template_gpu" $fname_gpu
cp "$dir/$template_cpu" $fname_cpu
loss='BW'
lr=1e-5
date=`date +%y%m%d`  # datestamp, format yymmdd
oroot=results/$date/ # root for the results 
name=loss-$loss

for fname in $fname_gpu $fname_cpu; do
    opt=$([[ $fname == $fname_gpu ]] && echo $opt_gpu || echo $opt_cpu)
    sed -i "s%^\(#SBATCH -J\) .*%\1 $sbname%" $fname
    sed -i "s%^\(#SBATCH -o ./slurm/out\).*%\1/$sbname-\%A_\%a.out%" $fname
    sed -i "s%^\(#SBATCH -e ./slurm/out\).*%\1/$sbname-\%A_\%a.err%" $fname
    echo "python train_linear.py \`sed -n \"\${SLURM_ARRAY_TASK_ID}{p;q}\" $opt\`" >> $fname
    [[ -f $opt ]] && rm $opt;
done

# for different configurations
for tau in 0 0.1 0.5 1; do
    for smin in -1 `seq 0.5 0.5 5`; do  # different sigma min
        for depth in `seq 1 6 9` `seq 10 5 100` ; do  # different depths

            [[ $depth > 80 ]] && USE_GPU=1  || USE_GPU=  # rule to use GPU

            # opt will either be gpu or cpu file 
            opt=$([[ $USE_GPU ]] && echo $opt_gpu || echo $opt_cpu)

            # write the options to the slurm file
            echo " --output $oroot --name $name --vary-name tau/d/smin -t $tau --smin $smin -d $depth -lr $lr -x $x -w $w -z $z  -T 3 --cvg-test --loss $loss"  >> $opt
        done;
    done;
done;

# launch the different jobs
for fname in $fname_cpu $fname_gpu; do
    opt=$([[ $fname == $fname_gpu ]] && echo $opt_gpu || echo $opt_cpu)
    [[ ! -s  $opt ]] && continue  # $opt is the file to consider, gpu or cpu. If empty go to the other 
    narray=`wc -l < $opt`  # number of array required

    sed -i "s%^\(#SBATCH --array=\?\).*$%\1 1-${narray}%" $fname  # set the number of batch array
    sbatch $fname
done


