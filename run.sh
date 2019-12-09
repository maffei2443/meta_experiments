#PBS -l select=1:ngpus=1
#PBS -l walltime=3:00:00

cd /home/$USER/meta_experiments/
python metastream_clf/datawragling.py
python metastream_clf/default-elec2.py --omega 300 --gamma 50 --initial 200 --target class --eval_metric acc > output.log
