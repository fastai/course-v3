---
title: Salamander
sidebar: home_sidebar
---

# Welcome to Salamander!

<img alt="" src="/images/salamander/logo.png" class="screenshot">

This guide takes about 1 minute to complete. Once complete, you will have access to a GPU-enabled Jupyter Notebook and the course-v3 materials

If you are returning to work and have previously completed the steps below, please go to the [returning to work](https://course.fast.ai/update_salamander.html) section.

## Pricing

We reccomend the "Accelerated Computing" configuration which costs **\$0.38 per hour**.

### Storage

You should use the suggested 75GB disk size, which is an **additional \$7.50 per month**. You can increase the disk size later if you need more space.

### Free credits for students

You can redeem any AWS coupon on Salamander. Students can claim between **$75 to $110 free credits** via the GitHub education pack.

### How much will you use this course

Considering that the course requires, over 2 months, 80 hours of homework plus the 2 hours of working through each lesson, we calculated roughly how much you would spend in the course.

- _Accelerated Computing_ + _Storage_: (80+2\*7)\*$0.38 + $7.50\*2 = **\$50.72**

## Step 1: Create an account

Visit [https://salamander.ai](https://salamander.ai), click "Get Started", complete the signup form, and add your card details.

<img alt="" src="/images/salamander/create_account.png" class="screenshot">

Students can claim their free credits at [https://salamander.ai/redeem-aws-coupon](https://salamander.ai/redeem-aws-coupon).

<img alt="" src="/images/salamander/coupon.png" class="screenshot">

## Step 2: Create your server

Make sure the "PyTorch 1.0, fastai, and the v3 MOOC course" software is selected, leave the remaining defaults as they are, and click "Launch server".

<img alt="" src="/images/salamander/create_server.png" class="screenshot">

Your server will appear straight away, and after a couple minutes the orange status indicator will turn blue.

<img alt="" src="/images/salamander/ready.png" class="screenshot">

## Step 3: Open Jupyter Notebook

Click 'Jupyter Notebook' to access the course materials. After Jupyter Notebook loads, click on 'course-v3'.

<img alt="nb tuto" src="/images/salamander/final.png" class="screenshot">

See [here](index.html) for instructions on running the Jupyter Notebook tutorial. Return to this guide once you're finished and _don't forget to stop your server_.

## Step 4: Update the fastai library

Before you start working you will need to update the fastai library. To do this you will have to access the terminal. You can do this by clicking in 'New', 'Terminal'.

<img alt="terminal" src="/images/terminal.png" class="screenshot">

Once you click on 'Terminal' a new window should open with a terminal. Type:

```bash
source activate fastai
conda install -c fastai fastai
```

Now you should close the terminal window.

## Step 5: Stop your server

When you're all done, **don't forget to shut down your server**, so you don't get charged for the time it's running in the background. It's not enough to just close your browser or turn off your own computer. Go back to [salamander](https://salamander.ai/) and click the 'Stop Server' button next to your server.

<img alt="" src="/images/salamander/stop.png" class="screenshot">

For more details, & updating the course / fastai library see "[Returning to work](update_salamander.html)".

## Advanced users: Full server access

Salamander isn't just for Jupyter Notebook - each server is a full Ubuntu instance. To connect, click "Setup Access Key" and generate or upload an ssh key - it'll get added to all of your servers automatically. You can then copy & paste `ssh ubuntu@[xxx.xxx.xxx.xxx]` from the webpage to your terminal. Press enter and you're in!

Note: you should always generate keys yourself if you'd like to use them for several different platforms ([in-depth guide](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/))

---

_Many thanks to Ashton Six for writing this guide._
