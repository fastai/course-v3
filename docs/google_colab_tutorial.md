---

title: Google Colab
keywords: 
sidebar: home_sidebar


---
# Welcome to Google Colab

[https://colab.research.google.com/](https://colab.research.google.com/)

Google Colab provides a free Jupyter Notebook environment that includes 4 NVIDIA Tesla K80 GPUs. 

## Pricing
Free

## Step 1: Navigate your browser to https://colab.research.google.com


## Step 2: Sign in to your Google account or create a new one 

After sign in Google will redirect you to a Colab notebook with information on how to load and save data from external sources, including GitHub and Google Drive.

## Step 3: Start a new Python 3 Colab notebook 
File->New Python 3 notebook
Click the title to rename it to whatever you like.

## Step 4: Set Runtime to GPU
Runtime->Change runtime type
Select GPU from the dropdown menu and click Save

## Step 4: Install PyTorch and FastAI with GPU support
You can install packages with !pip install

See the Colab Notebook at https://colab.research.google.com/drive/1WMc4bO2NnFPS4ME8EIo_Z6xLyEh94Vf2 or use the code snippet below:

```
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag

platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

!pip install torch_nightly -f https://download.pytorch.org/whl/nightly/{accelerator}/torch_nightly.html
!pip install fastai

import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)
```

It should take a minute or two to install PyTorch and FastAI. Then you should see output that starts with 1.0 and True True

```
import fastai
print(fastai.__version__)
from fastai import *
from fastai.vision import *
```

You should see output that starts with 1.0


## Step 5: 
You can easily save land load files from GitHub and Google Drive.

Troubleshooting
Some users get only 5% of the available memory. If that's you, sorry, that stinks. Hopefully this is something that Google fixes soon.

Set num_workers=0 if you are getting an error similar to: 'RuntimeError: DataLoader worker (pid 216) is killed by signal: Bus error.' 
For example in 'data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), tfms=imagenet_norm, size=224, num_workers=0) if having issues with running out of memory.'

## Other Useful Info

Instances can run for 12 hours. You will need to rerun your files after that. 

You won't see a nice progress bar if you are in Colab because it doesn't support Ipywidgets. You can throw your support behind the issue here: https://github.com/googlecolab/colabtools/issues/60
