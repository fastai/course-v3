---
title: Colab
keywords: 
sidebar: home_sidebar
---

This is a quick guide to starting v3 of the fast.ai course Practical Deep Learning for Coders using Colab. 

If you are returning to work and have previously completed the steps below, please go to the [returning to work](http://course-v3.fast.ai/update_colab.html) section.

**NB: This is a free service that may not always be available, and requires extra steps to ensure your work is saved. Be sure to read the docs on the Colab web-site to ensure you understand the limitations of the system.**

## Getting Set Up

### Step 1: Accessing Colab

1. First of all you should sign in to you Google account if you are not signed in by default. You must do this step before opening Colab, otherwise the notebooks will not work. You can sign in [here](https://accounts.google.com/signin/v2/identifier?hl=en-gb&flowName=GlifWebSignIn&flowEntry=ServiceLogin).

1. Next, head on to the [Colab Welcome Page](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true) and click on 'Github'. In the 'Enter a GitHub URL or search by organization or user' line enter 'fastai/course-v3'. You will see all the courses notebooks listed there. Click on the one you are interested in using.

    <img alt="stop" src="/images/colab/01.png" class="screenshot">

1. You should see your notebook displayed now. Before running anything, you need to tell Colab that you are interested in using a GPU. You can do this by clicking on the 'Runtime' tab and selecting 'Change runtime type'. A pop-up window will open up with a drop-down menu. Select 'GPU' from the menu and click 'Save'.

    <img alt="stop" src="/images/colab/03.png" class="screenshot">

    <img alt="stop" src="/images/colab/04.png" class="screenshot">


### Step 2: Configuring your notebook instance

1. Before you start using your notebook, you need to install the necessary packages. You can do this by creating a code cell, and running:

    ```bash
     !curl https://course-v3.fast.ai/setup/colab | bash
    ```

    <img alt="create" src="/images/colab/05.png" class="screenshot">

1. If you a face a pop-up saying 'Warning: This notebook was not authored by Google' you should leave the default tick in the 'Reset all runtimes before running' check box and click on 'Run Anyway'.

    <img alt="stop" src="/images/colab/02.png" class="screenshot">

1. On the new window click 'Yes'.

    <img alt="stop" src="/images/colab/08.png" class="screenshot">

1. Delete, if any, cells that contain:

    ```bash
    %reload_ext autoreload
    %autoreload 2
    %matplotlib inline
    ```

### Step 3: Saving your work

If you opened a notebook from Github, you will need to save your work to Google Drive. You can do this by clicking on 'File' and then 'Save'. You should see a pop-up with the following message:

<img alt="create" src="/images/colab/09.png" class="screenshot">

Click on 'SAVE A COPY IN DRIVE'. This will open up a new tab with the same file, only this time located in your Drive. If you want to continue working after saving,  use the file in the new tab.

## More help

For questions or issues related to course content, we recommend posting in the [fast.ai forum](http://forums.fast.ai/).

