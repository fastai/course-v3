---
title: DataCrunch.io
keywords: datacrunch
sidebar: home_sidebar
---

# Welcome to DataCrunch.io

[DataCrunch.io](https://datacrunch.io) offers the lowest-cost dedicated NVidia Tesla V100 servers. You can choose a Fast.ai image, which contains all dependencies for Fast.ai v1 and v2.

## Pricing

The suggested server for fast.ai is a [1V100.6V image](https://datacrunch.io/products/), which has 1 dedicated NVidia Tesla V100 GPU with 6 CPU cores. The instance will cost $0.52/h when using the FastAI discount code **FastAI20%** upon topping up.

## Step 1: Signing up

Create your account on [cloud.datacrunch.io](https://cloud.datacrunch.io/signup). Verify your e-mail address to complete creating your account.

## Step 2: Top up your account

To prevent you from getting charged more than planned, we use a prepaid system. You can only use the amount loaded onto your account.
To deploy a server, you will need to top-up an amount first. Add a credit card to your account and top-up after adding the coupon code.

<img alt="AddCredit" src="/images/datacrunch/add_credit.png" class="screenshot">

## Step 3: Create an SSH key.

You will need an SSH key to allow SSH access to your instance. A key can be generated using command line tools or tools which provide a convenient user interface such as [PuttyGen](https://www.puttygen.com/).

0. Launch PuttyGen.
1. Choose Ed25519. It's the most recommended public-key algorithm available today.
2. Click generate and move your cursor in the empty box to create some additional randomness.
3. Copy the generated key to your clipboard, you will need it later.
4. Save your private key somewhere safe.

<img alt="PuttyGen" src="/images/datacrunch/puttygen.png" class="screenshot">

## Step 4: Deploy your instance

After logging in on and topping up your account, you can launch a server by clicking [Deploy a New Server](https://cloud.datacrunch.io/dashboard/deploy-server).

1. Choose the 1V100.6V instance
2. Choose the Fast.ai image
3. Click 'Add Key'
4. Name your key and paste the public key you generated in step 3.
5. Click 'Create' to add your key to your account.
6. Choose a hostname and description for your instance.
7. Click 'Deploy'.

<img alt="Deploy DataCrunch.io server" src="/images/datacrunch/deploy_server.png" class="screenshot">

## Step 5: Launch Jupyter Notebook

About 1 minute after deploying, your server is ready. When the status icon is 'running', click 'Open Juypter Notebook' to open Jupyter Notebook.

<img alt="Launch Jupyter Notebook" src="/images/datacrunch/launch_jupyter.png" class="screenshot">

## Step 6: Load your course material

You can 'git clone' your desired repo, either by using your SSH key for username 'user', by launching a terminal from Jupyter Notebook or by entering a command in a jupyer notebook (By executing a cell like '!git clone mygitlink.git'). Let's launch a terminal from your notebook server:

<img alt="Jupyter terminal" src="/images/datacrunch/jupyter_terminal.png" class="screenshot">

'git clone' your desired repository, for fastai we run 'git clone https://github.com/fastai/fastai.git'

<img alt="Jupyter terminal git" src="/images/datacrunch/jupyter_terminal_git.png" class="screenshot">

## Step 7: Removing your instance

In order to stop your instance from decreasing your balance, you need to delete it.
We are adding the option to store your state, at the moment deleting is the only option.

Save your notebooks to your PC if you desire to work on them later:

<img alt="Save Notebook" src="/images/datacrunch/save_notebook.png" class="screenshot">
