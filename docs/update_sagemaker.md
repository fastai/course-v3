---
title: Returning to Sagemaker
keywords: 
sidebar: home_sidebar
---

To return to your notebook, the basic steps will be:

1. Start your instance
1. Update the course repo
1. Update the fastai library
1. When done, shut down your instance

## Step by step guide

### Start your instance

Log in to the [AWS console](https://aws.amazon.com/console/) then click on the Sagemaker link (it should be in your history, otherwise find it in the 'Services' on the left or type sagemaker in the search bar). Once on this page, select 'Notebook instances' on the left menu.

<img alt="" src="/images/sagemaker/notebooks.png" class="screenshot">

Tick the box of the notebook you want to start, then click on 'Start'.

<img alt="" src="/images/sagemaker/start.png" class="screenshot">


You will have to wait a little bit for your instance to be ready while the light under instance state is orange.

<img alt="pending" src="/images/sagemaker/16.png" class="screenshot">

When it turns green, just click on 'Open' and you'll be back to your notebooks.

<img alt="ready" src="/images/sagemaker/17.png" class="screenshot">

### Update the course repo
To update the course repo, launch a new terminal from the jupyter notebook menu.

<img alt="" src="/images/gradient/terminal.png" class="screenshot">

This will open a new window, in which you should run those two instructions:

``` bash
cd SageMaker/course-v3
git pull
``` 

<img alt="" src="/images/gradient/update.png" class="screenshot">

This should give you the latest of the course notebooks. If you modified some of the notebooks in course-v3/nbs directly, GitHub will probably throw you an error. You should type `git stash` to remove your local changes. Remember you should always work on a copy of the lesson notebooks.

### Update the fastai library
To update the fastai library, open the terminal like before and type
``` bash
source activate SageMaker/envs/fastai
conda install -c fastai fastai
```
Note that you have to be in the home directory (the one the terminal puts you in when you create it) for this to work.

### Stop your instance
When you finish working you must go back to your [AWS console](https://us-west-2.console.aws.amazon.com/sagemaker) and stop your instance manually to avoid getting extra charges. Just pick the notebook you want to stop and click on the 'Stop' button next to its name.

<img alt="stop" src="/images/sagemaker/23.png" class="screenshot">

 **NOTE: you *will* be charged for the time that your instance is running.**

