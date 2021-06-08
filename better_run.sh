#PBS -l select=1:ngpus=1
#PBS -l walltime=3:00:00

N_RUNS=$1
cd metastream

# echo incremental
# for i in 'elec2';# 'powersupply' 'covtype';
#   do
#   echo $i
#   echo start: $(date "+%y%m%d.%H%M%S.%3N")
#   python ms_incremental.py --omega 300 --gamma 20\
#       --initial 300 --target class --eval_metric acc\
#       --path data/$i/ > $i.output
#   echo stop:  $(date "+%y%m%d.%H%M%S.%3N")
# done

echo non incremental
echo $N_RUNS0

mkdir experiments_outputs -p

for ((i=1; i<=N_RUNS; i++))  do
    echo "N_RUN: $i"
    for dataset in 'elec2'; do
      echo "  Run: $i, data:  $dataset"
      echo "    $i-ninc"
      echo "    start:" + $(date "+%y%m%d.%H%M%S.%3N")
      python ms_nonincremental.py --omega 300 --gamma 20\
          --initial 300 --target class --eval_metric acc\
          --path data/${dataset}_ninc/ > experiments_outputs/$dataset-ninc-$i.output
      echo stop:  $(date "+%y%m%d.%H%M%S.%3N")
    done
  done
  # for dataset in 'elec2';# 'powersupply' 'covtype';
  #   echo "Run: $i, data:  $dataset"
  #   do
  #     echo $i-ninc
  #     echo start: $(date "+%y%m%d.%H%M%S.%3N")
  #     python ms_nonincremental.py --omega 300 --gamma 20\
  #         --initial 300 --target class --eval_metric acc\
  #         --path data/${dataset}_ninc/ > $dataset-ninc.output
  #     echo stop:  $(date "+%y%m%d.%H%M%S.%3N")


# echo streams

# echo incremental
# for i in 'hyper' 'rbf'; # 'agrawal';
#   do
#   echo $i
#   echo start: $(date "+%y%m%d.%H%M%S.%3N")
#   python metastream/ms_incremental.py --omega 300 --gamma 20 --initial 300 --target class --eval_metric acc --path data/$i/ > $i.output
#   echo stop:  $(date "+%y%m%d.%H%M%S.%3N")
# done

# echo non incremental
# for i in 'hyper' 'rbf'; # 'agrawal';
#   do
#   echo $i-ninc
#   echo start: $(date "+%y%m%d.%H%M%S.%3N")
#   python metastream/ms_nonincremental.py --omega 300 --gamma 20 --initial 300 --target class --eval_metric acc --path data/${i}_ninc/ > $i-ninc.output
#   echo stop:  $(date "+%y%m%d.%H%M%S.%3N")
# done
