---
title: Gradient
sidebar: home_sidebar
---
# Fast.ai Deep Learning Course v3 on Gradient° Notebooks

<img alt="" src="/images/gradient/gradientFastAIv3.png" class="screenshot">

This is a quick guide to starting v3 of the Fast.ai course. With [Gradient](https://www.paperspace.com/gradient), you get access to a Jupyter Notebook instance without complicated installs or configuration in less than 2 minutes.

[Gradient](https://www.paperspace.com/gradient) is built on top of [Paperspace](https://www.paperspace.com/), a GPU-accelerated cloud platform. 

If you are returning to work and have previously completed the steps below, please go to the [returning to work](https://course.fast.ai/update_gradient.html) section.

## Pricing

Notebooks are billed while they're running (per second!) and the rate is dependent on the [Instance Type](https://support.paperspace.com/hc/en-us/articles/360007742114-Gradient-Instance-Types) selected.  Notebooks must be stopped to end billing. See below for free GPU credit! 💰

## Step 1: Create an account
To get started, create an account [here](https://www.paperspace.com/account/signup) and confirm your account by clicking the verification link in your inbox.

## Step 2: Create Notebook
1. Login and select Gradient > Notebooks.

2. Select the *Paperspace + Fast.AI 1.0 (V3)* base container.

<img alt="" src="/images/gradient/createNotebook.png" class="screenshot">

3. Select the type of machine you want to run on.

<img alt="" src="/images/gradient/chooseMachineType.png" class="screenshot">

4. Name your Notebook (optional)

5. Enter your payment details (if you're new to Paperspace). *Even if you have a promo or referral code, all active Paperspace accounts must have a valid credit card on file. You'll be able to enter your promo code later.*

6. Click Create Notebook

   <img alt="create" src="/images/gradient/create.png" class="screenshot">

Your Notebook will go from Pending to Running, and will be ready to use :star2:.

When you click Create Notebook, that will start your Notebook and your billing for utilization will begin. To stop billing, you must stop your Notebook. Notebooks will automatically shut down after 12 hours.

## Step 3 : Update the fastai library

Before you start working you will need to update the fastai library. To do this you will have to access the terminal. You can do this by clicking in 'New', 'Terminal'.

<img alt="terminal" src="/images/terminal.png" class="screenshot">

Once you click on 'Terminal' a new window should open with a terminal. Type:

``` bash
pip install fastai --upgrade 
```

Now you should close the terminal window.

## Step 4: Start learning Fast.ai!
You should now have a running fast.ai notebook. It might take a few minutes to provision, but once it's running you just have to click "Open" to access your Jupyter notebook.

<img alt="ready" src="/images/gradient/ready.png" class="screenshot">

Next from your your jupyter notebook, click on 'course-v3' and you should look at something like this:

<img alt="nb tuto" src="/images/jupyter.png" class="screenshot">

Go back to the [first page](index.html) to see how to use this jupyter notebook and run the jupyter notebook tutorial. Come back here once you're finished and *don't forget to stop your instance* with the next step

## Step 5: Stopping your Notebook
Just click stop.  This will end the billing session.

<img alt="" src="/images/gradient/stopNotebook.png" class="screenshot">

NOTE: you *will* be charged for the time that your notebook is running. You must stop the notebook to stop incurring charges.

For more details, updating the course and the fastai library see "[Returning to work](update_salamander.html)".

---

## Additional considerations:

### Managing Data
The `/storage` folder is your [Persistent Storage](https://support.paperspace.com/hc/en-us/articles/360001468133-Persistent-Storage). Files placed here are available across all Notebooks, Jobs, and Linux VMs (currently free of charge). This repository is perfect for storing datasets, models etc. Note: Persistent Storage is region specific (you'll see the storage region options when creating Notebooks and Jobs).

### Promotional credit
Paperspace provides $10 of free Gradient° credit. This code is to be used for Fast.ai students only. In your console, click on Billing and enter the promo code at the bottom right. The promo code for this course is: **FASTAIGR19**. 
*Note: the code is valid until Jan 1, 2020*

Note: If you opt for a Gradient 1 Subscription, promotional credit does not apply. [Learn more about Gradient Subscription levels here](https://support.paperspace.com/hc/en-us/articles/360002068913-Gradient-Subscriptions).

### Where to get help

Questions or issues related to course content, we recommend posting in the [fast.ai forum](http://forums.fast.ai/).

For Paperspace-specific support, check out the rest of the [Gradient Help Center](https://support.paperspace.com/hc/en-us/categories/115000426054-Gradient-) or submit a support ticket with [this form](https://support.paperspace.com/hc/en-us/requests/new).

---

*Many thanks to [Dillon Erb](https://github.com/dte) for writing this tutorial*
