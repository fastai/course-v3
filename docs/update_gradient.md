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

_**Each time that you start your notebook, you can choose a different virtual machine type on which it runs.**_ Make sure you're running it on the  machine you want! Prices vary enormously. This can be extremely useful when you want to start on a lower-end machine type, test that everything is okay, then move to a more powerful GPU. Also, sometimes the GPU type that you started the notebook on will be unavailable, in which case you can easily fire it up on a different GPU.

When you click that 'start' button, the following screen will appear. It should default to the last machine type you used, but it's a good idea to check!

<img alt="" src="/images/gradient/selectedMachine.png" class="screenshot">

If you want to change the machine type, select from the dropdown:

<img alt="" src="/images/gradient/changeMachineType.png" class="screenshot">


 (See [Gradient pricing](https://support.paperspace.com/hc/en-us/articles/360002484474-Gradient-Pricing) for more information.) 



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