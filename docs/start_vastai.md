---

title: vast.ai 
keywords: 
sidebar: home_sidebar


---
# Welcome to vast.ai beta

[https://vast.ai](https://vast.ai)

vasta.ai provides access to GPUs on third party machines. Data may not be private or secure. See discussion here: https://vast.ai/faq/

## Pricing
Varies based on supply and demand.

10GB storage is included in the price shown (it's $.002/hr, as you see if bidding for a server)

It costs $.02/GB to upload and download data.

You get $2.00 of credit when you enter your payment info.

## Step 1: Navigate your browser to https://vast.ai/console/create/


## Step 2: Create an account 

Fill out the form and check your email to confirm signup.

## Step 3: Choose your image
Click *Select Image*.  
Choose *pytorch/pytorch*.
For *version tag to use* choose the one that starts with *nightly_devel_cuda9.2*

## Step 4: Choose whether to rent or bid.
Accept a price and start by pressing _Rent_ on the machine of your choice.
Or make a _Make Bid_ to bid on a machine. If you a make a bid and then someone rents the machine at the full price, you loose out. 
Renting leads to less frustration than bidding.

If a machine has an AWS, Google Cloud, or Paperspace logo, then clicking *Rent* will just take you to that provider's website.

## Step 5: Enter your payment info.
You will be asked to setup your payment info. 

Payment info from vast.ai: 
*Billed every Friday. You won't be charged until your credits run out. We use Stripe , so your card is never sent to our servers.*

## Step 6: Start instance
It will take a few minutes to load.
Click on *Instances* in the menu on the left. 
Click on *Start* on your instance
Click on *Connect* when the button changes to *Connect*


## Step 7: Your JupyterNotebook or JupyterLab should automatically open on your browser at "https://jupyter.vast.ai/..."

Install fastai and you're off!
```
!pip install fastai

import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)
```

It should take a minute or two to install fastai. Then you should see output that starts with 1.0 and True True

```
import fastai
print(fastai.__version__)
from fastai import *
from fastai.vision import *
```

You should see output that starts with 1.0

Proceed to code. 

## Don't forget to click stop on your instance at vast.ai -> Instances or you will continue to be billed!
