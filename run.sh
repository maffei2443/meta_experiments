#PBS -l select=1:ngpus=1
#PBS -l walltime=3:00:00

for i in 'elec2' 'aws' 'powersupply' 'covtype';
  do
    echo $i
    echo start: $(date "+%y%m%d.%H%M%S.%3N")
    python -W ignore metastream/ms_incremental.py --omega 300 --gamma 10 --initial 300 --target class --eval_metric acc --path data/$i/ > $i.output
    echo stop:  $(date "+%y%m%d.%H%M%S.%3N")
done
echo non incremental
for i in 'elec2' 'aws' 'powersupply' 'covtype';
  do
    echo $i
    echo start: $(date "+%y%m%d.%H%M%S.%3N")
    python -W ignore metastream/ms_nonincremental.py --omega 300 --gamma 10 --initial 300 --target class --eval_metric acc --path data/${i}_ninc/ > $i-ninc.output
    echo stop:  $(date "+%y%m%d.%H%M%S.%3N")
done
