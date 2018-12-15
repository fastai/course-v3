#!/bin/bash
#set -e
set -x
# create symlinks if not existing
echo "Creating symlinks"
[ ! -L "/home/ec2-user/.torch" ] && ln -s /home/ec2-user/SageMaker/.torch /home/ec2-user/.torch
[ ! -L "/home/ec2-user/.fastai" ] && ln -s /home/ec2-user/SageMaker/.fastai /home/ec2-user/.fastai

# check if the onCreate script has finished running
if [ -f "/home/ec2-user/.fastai/create_complete" ]; then
    echo "Running the standard start script"
    /home/ec2-user/SageMaker/.fastai/std-start-script.sh

    echo "Running the custom start script"
    /home/ec2-user/SageMaker/custom-start-script.sh
else
    echo "Still running create script..Exiting"
fi
echo "Finished running onStart script"

