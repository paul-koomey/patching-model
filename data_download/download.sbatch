#!/bin/bash
#SBATCH --job-name=test_download_20 # Job name
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu    # Where to send mail
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=16                   # Run a single task	
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=32gb                  # Total memory limit
#SBATCH --time=96:00:00              # Time limit hrs:min:sec
#SBATCH --output=download.log     # Standard output and error log

date; hostname; pwd

echo "Download Started!"

gdc-client download -m ~/projects/vae/data/download_manifest.txt --config ~/projects/vae/data/my-dtt-config.dtt

echo "Download Finished!"

date
