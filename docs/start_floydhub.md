---
title: FloydHub
sidebar: home_sidebar
---
# Fast.ai Deep Learning Course v3 on FloydHub

<img alt="" src="/images/floydhub/floydhubFastai.png">

This is a quick guide to starting v3 of the Fast.ai course. With [FloydHub](https://www.floydhub.com), you get access to a dedicated JupyterLab instance without complicated installs or configuration, all in less than 2 minutes. In addition to one-click Jupyter notebooks, FloydHub also supports long-running model-training jobs and easy-setup model-serving APIs.

If you are returning to work and have previously completed the steps below, please go to the [returning to work](http://course-v3.fast.ai/update_floydhub.html) section.

## Pricing

FloydHub Jupyter workspaces and jobs are billed while they're running (per second!) and the rate is dependent on the [Machine](https://docs.floydhub.com/faqs/plans/#what-are-powerups) selected. You can get access to different machines by purchasing Powerups.

The Beginner plan on FloydHub is free, and comes with 20 hours of free CPU Powerups per month. Also, if you add your credit card, you'll receive a free 2-Hour GPU (K80) powerup.

Workspaces must be stopped to end billing, but FloydHub will automatically shut down your Workspace after 15 minutes of inactivity. By default, this is set to 15 minutes, but you can [easily adjust this setting](https://docs.floydhub.com/guides/workspace/#idle-timeout-detection). See below for free GPU credit! 💰

## Step 1: Click this Run on FloydHub button

[![Run on FloydHub](https://static.floydhub.com/button/button-small.svg)](https://floydhub.com/run?template=https://github.com/fastai/course-v3)

FloydHub's goal is to make the process of deploying model APIs, build models in Jupyter notebooks, and running model-training scripts simple, easy, and reproducible. The "Run on FloydHub" button is a simple HTML / Markdown snippet that can be added to READMEs, blog posts, and other places where your code lives.

Clicking this FloydHub button takes you through a guided process that sets up a Jupyter Workspace with the Fast.ai course-v3 source code.

## Step 2: Create an account

First, you'll be asked to create your FloydHub account. You can also use your Google or GitHub accounts to sign up and log-in. Be sure to confirm your FloydHub account by clicking the verification link in your inbox.

## Step 3: Create your first Project

Next, you'll be redirected back to the `Create Project from Repo` page.

<img alt="" src="/images/floydhub/createProject.png" class="screenshot">

A [FloydHub config file](https://github.com/fastai/course-v3/floyd.yml) has been added to the `fastai/course-v3` repo, so FloydHub will adjust your Workspace's Environment (aka Docker image) to PyTorch-1.0. You'll also notice that the Machine is initially set to CPU. We'll show you how to toggle your Workspace to GPU mode in Step 5.

## Step 4: Run your first Workspace with a CPU Powerup

After creating your project, you'll be directed to the project's list of Workspaces.

<img alt="" src="/images/floydhub/resumeWorkspace.png" class="screenshot">

Click the "Resume" link, and then click the name of the Workspace to open up your Workspace.

That's it! You're now in your first Workspace on FloydHub. Behind the scenes, FloydHub has set up your machine with PyTorch-1.0, the latest `fastai` library, and many other useful machine learning packages. You can think about this Workspace as an interactive development environment, featuring support for multiple Jupyter Notebooks, writing Python scripts, running terminal sessions, and easy file navigation.

<img alt="" src="/images/floydhub/workspace.png" class="screenshot">

## Step 5: Restart your Workspace using GPU Powerup

Now that you've got a running Workspace on a CPU machine, let's restart it using a GPU machine.

**Click the "Restart" button** in the upper right top-nav bar of your Workspace. You'll see a dropdown list of machine-types. Once you have available hours, these options will be filled in and let you easily toggle between machines. Let's add your free 2 Hour GPU Powerup now. Under "GPU", click `Add more hours`.

<img alt="" src="/images/floydhub/restart.png" class="screenshot">

You'll be navigated to your [Settings-Powerups](https://floydhub.com/settings/powerups) page, where you can manage and purchase more Powerup hours as needed. This is a good page to know about! FloydHub accepts credit cards or PayPal for Powerup purchases. On the top of this page, you should see a banner that says: `Claim your free 2-Hour GPU powerup now`. **Click that banner link, and fill out your Payment info.**

<img alt="" src="/images/floydhub/payment.png" class="screenshot">

Now you should see a new 2 Hour GPU powerup in your Settings page.

Go back to your running Workspace and refresh the page. You'll now be able to click Restart button and select the GPU option.

<img alt="" src="/images/floydhub/restart.gif" class="screenshot">

You're now live in a GPU-powered Workspace on FloydHub.

## Step 6: Stopping your Workspace
Just click the Shutdown button. This will end the billing session. FloydHub will automatically shut down your Workspace once we detect 15 minutes of inactivity (you can adjust this setting!), and store your code for your next Workspace session. You can navigate and view your code while your Workspace is Shutdown.

<img alt="" src="/images/floydhub/shutdown.png" class="screenshot">

NOTE: you *will* be charged for the time that your Workspace is running. You must shutdown the Workspace to stop incurring charges.

For more details, updating the course and the fastai library see "[Returning to work](update_floydhub.html)".

---

## Additional considerations:

### Persisting data
Anything you save in the `/floyd/home` directory will be persisted, which means you'll have access to these files every time you Restart your Workspace. FloydHub also adjusted the default fastai config.yml file to store your datasets and models in the `/floyd/home/fastai/data` directory. This helps you navigate your datasets in the Workspace's file-browser, and will also persist your datasets and models each time you open your Workspace.

### Promotional credit
FloydHub provides 20 hours of free CPU per month, as well as a free 2-Hour GPU Powerup when you enter your credit card details.

### Where to get help

Questions or issues related to course content, we recommend posting in the [fast.ai forum](http://forums.fast.ai/).

For FloydHub-specific support, check out the rest of the [FloydHub forum](https://forum.floydhub.com), reach out on Twitter at [@floydhub_](https://twitter.com/floydhub_), or [email FloydHub support](mailto://support@floydhub.com).

---

*Many thanks to [Charlie Harrington](https://twitter.com/whatrocks) of Group 23 for writing this tutorial*