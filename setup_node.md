## Template for the Sbatch file

#!/bin/bash

#SBATCH --job-name=train
#SBATCH --partition=batch                                           	# only the batch partition
#SBATCH --nodes=2                                                  	     # maximum 5 for us in total
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --time=4-00:00:00                                                  #maximum 4 for now
#SBATCH --output=log/test%j.txt
#SBATCH --error=log/error%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user= xxx

module load slurm
module load cuda/12.1
source /home/lixiaomeng/miniconda3/etc/profile.d/conda.sh
conda activate xxx

salloc -J train -N 1 -n 4 -p batch --time=05:00:00 --gres=gpu:2


## Directly debug on the GPU

1. Get the GPU resource

```
salloc -J jobname -N 1 -n 24 -p batch --time=05:00:00 --gres=gpu:2
```

2. Run on the GPU

```
srun --jobid xx -w dgx-xx --pty bash
```

3. Run your Code😊



### Upload Files to Slogin Server

1. From local to slogin

```
rsync -av --progress --partial -e 'ssh -p 22 -J lixiaomeng@143.89.224.47' FILE_PATH lixiaomeng@10.33.4.51:TARGET_PATH
```

2. From jumpserver to slogin:

```
rsync -av --progress --partial FILE_PATH slogin:TARGET_PATH
```



### NOTES

1. **Save large files and the checkpoints on the /aifs4su/lixiaomeng**
2. **Maximum task numbers are 6 for us**
3. **Any modifications of the code would influence the queued task**