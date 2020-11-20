
#$ -cwd -V
#$ -l h_rt=48:00:00
#$ -l coproc_p100=2
#$ -M cnlp@leeds.ac.uk
#$ -m be

echo "Starting C3D+MIME training script..."

module load python
module load cuda
module load python-libs/3.1.0

python < C3D_MIME20_train_90_10_shuffle.py >> C3D_MIME20_train_90_10_shuffle.txt
