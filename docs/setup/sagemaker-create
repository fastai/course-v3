#!/bin/bash
#set -e
set -x

# create symlinks to EBS volume
echo "Creating symlinks"
mkdir /home/ec2-user/SageMaker/.torch && ln -s /home/ec2-user/SageMaker/.torch /home/ec2-user/.torch
mkdir /home/ec2-user/SageMaker/.fastai && ln -s /home/ec2-user/SageMaker/.fastai /home/ec2-user/.fastai

# install the fastai library and dependencies
echo "Install the fastai library and dependencies in new conda enviornment"
cd /home/ec2-user/SageMaker
# for explanations of the -mqyp flags, check `conda create --help`
conda create -mqyp envs/fastai -c pytorch -c fastai fastai ipykernel
echo "Finished installing environment"

# clone the course notebooks
echo "Clone the course repo"
git clone https://github.com/fastai/course-v3.git /home/ec2-user/SageMaker/course-v3

# install a custom start script which can be modified
echo "Install the custom start script"
cat > custom-start-script.sh <<\EOF
#!/bin/bash
if [ ! "$CONDA_DEFAULT_ENV" = "fastai" ]; then
    echo "Activate fastai conda environment"
    cd /home/ec2-user/SageMaker
    source activate envs/fastai
fi
echo "Updating fastai library"
conda install -y -c fastai fastai
echo "Getting latest version of course"
cd /home/ec2-user/SageMaker/course-v3
git pull
echo "Finished running custom start script"
EOF
chmod u+x custom-start-script.sh

# install a standard start script which is run each time the instance is started
echo "Install the standard start script"
cat > /home/ec2-user/SageMaker/.fastai/std-start-script.sh <<\EOF
#!/bin/bash
# install the ipython kernel
echo "Install the ipython fastai kernel"
cd /home/ec2-user/SageMaker
source activate envs/fastai

cuda_version=$(nvcc --version | awk 'match($0, /Cuda compilation tools, release [0-9]+\.[0-9]+/) { print substr($0, RSTART+32, RLENGTH-(RSTART+31))}')
echo "Found Cuda version $cuda_version"
# Install pytorch with correct cuda version
# This command comes from the "Quick Start Locally" tool at https://pytorch.org/ as of 12 February, 2019
# With selections Stable (1.0) / Linux / Conda / Python 3.6 / CUDA 9.0
conda install -y pytorch torchvision cudatoolkit=${cuda_version} -c pytorch

pip install --upgrade sagemaker

python -m ipykernel install --name 'fastai' --display-name 'Python 3' --user
echo "Install jupyter nbextension"
source /home/ec2-user/anaconda3/bin/activate JupyterSystemEnv
pip install jupyter_contrib_nbextensions
jupyter contrib nbextensions install --user
echo "Restarting jupyter notebook server"
pkill -f jupyter-notebook
EOF
chmod u+x /home/ec2-user/SageMaker/.fastai/std-start-script.sh

# put a empty file so start script knows if already run create script
touch /home/ec2-user/.fastai/create_complete

echo "Running standard start script"
/home/ec2-user/SageMaker/.fastai/std-start-script.sh

echo "Finished running onCreate script"
