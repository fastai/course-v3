---
title: Returning to Gradient
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

In your [console notebooks](https://www.paperspace.com/console/notebooks) choose the notebook you want to run and click on the button 'Start' under action.

<img alt="" src="/images/gradient/start.png" class="screenshot">

You can choose a different Virtual Machine type on which you'd like to run your Notebook. This can be extremely useful when you want to start on a lower-end machine type, test everything is okay, then move to a more powerful GPU. Also, sometimes the GPU type that you started the notebook on will be unavailable, in which case you can easily fire it up on a different GPU.

<img alt="" src="/images/gradient/restartNotebook.png" class="screenshot">

Click on 'Start notebook' when you're ready and wait a few seconds for it to be ready.

<img alt="" src="/images/gradient/ready.png" class="screenshot">

Click on the open button when it's ready and you'll be back in your jupyter notebook page.

### Update the course repo
 To update the course repo, launch a new terminal from the jupyter notebook menu.

<img alt="" src="/images/gradient/terminal.png" class="screenshot">

This will open a new window, in which you should run those two instructions:

``` bash
cd course-v3
git pull
``` 

<img alt="" src="/images/gradient/update.png" class="screenshot">

This should give you the latest of the course notebooks. If you modified some of the notebooks in course-v3/nbs directly, GitHub will probably throw you an error. You should type `git stash` to remove your local changes. Remember you should always work on a copy of the lesson notebooks.

### Update the fastai library
To update the fastai library, open the terminal like before and type
``` bash
pip install fastai --upgrade 
```

### Stop your instance
Once you're finished, go back to your [console notebook page](https://www.paperspace.com/console/notebooks) and find your running notebook. Under Action, click stop, this will end the billing session.

<img alt="" src="/images/gradient/stopNotebook.png" class="screenshot">

 **NOTE: you *will* be charged for the time that your notebook is running. You must stop the notebook to stop incurring charges.**