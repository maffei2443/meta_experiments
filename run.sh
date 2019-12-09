#PBS -l select=1:ngpus=1
#PBS -l walltime=3:00:00

echo
echo start: $(date "+%y%m%d.%H%M%S.%3N")
echo

cd /home/$USER/meta_experiments/
# python metastream_clf/datawragling.py
python metastream_clf/default-elec2.py --omega 300 --gamma 50 --initial 200 --target class --eval_metric acc > output.log

echo
echo stop:  $(date "+%y%m%d.%H%M%S.%3N")
echo
