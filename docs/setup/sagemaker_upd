#!/bin/bash
cd /home/ec2-user/SageMaker
source activate envs/fastai
conda install -y -c fastai fastai
ipython kernel install --name 'fastai' --display-name 'Python 3' --user
cd course-v3
git stash
git pull
