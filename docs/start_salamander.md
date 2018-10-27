---

title: Salamander
sidebar: home_sidebar

---

# Welcome to Salamander!

It takes about 1 minute to signup & launch a Salamander server. The servers include everything you need to complete the fastai v3 course. Once launched, you can jump straight to Jupyter Notebook or connect directly via ssh.

If you are returning to work and have previously completed the steps below, please go to the [returning to work](http://course-v3.fast.ai/update_salamander.html) section.

## Pricing

Salamander tracks the AWS _spot_ price +26%. Prices at time of writing:

- K80: $0.36 per hour
- V100: $1.32 per hour

## Step 1: Create an account

Visit [https://salamander.ai](https://salamander.ai), click "Get Started", fill-in the form & add your card details.

<img alt="" src="/images/salamander/create_account.png" class="screenshot">

## Step 2: Create your server

> If you already have a Salamander account, we recommend creating a brand new server to get the latest version of fastai

Pick your desired hardware & storage size (if you don't know what to choose, just keep the default options). Don't forget to accept the 'cuDNN Software License Agreement' and to check the acknowledgements above "Launch Server" before clicking it.

<img alt="" src="/images/salamander/create_server.png" class="screenshot">

Wait about a minute for the server to start. Once finished, it will look like this:

<img alt="" src="/images/salamander/ready.png" class="screenshot">

## Step 3: Open Jupyter Notebook

Click 'Jupyter Notebook' to access the course materials. After Jupyter Notebook loads, click on 'fastai-courses' & then 'course-v3'.

<img alt="nb tuto" src="/images/salamander/final.png" class="screenshot">

See [here](index.html) for instructions on running the Jupyter Notebook tutorial. Return to this guide once you're finished and _don't forget to stop your server_.

## Step 4: Update the fastai library

Before you start working you will need to update the fastai library. To do this you will have to access the terminal. You can do this by clicking in 'New', 'Terminal'.

<img alt="terminal" src="/images/terminal.png" class="screenshot">

Once you click on 'Terminal' a new window should open with a terminal. Type:

``` bash
source activate fastai
conda install -c fastai fastai
```

Now you should close the terminal window.

## Step 5: Stop your server

When you're all done, **don't forget to shut down your server**, so you don't get charged for the time it's running in the background. It's not enough to just close your browser or turn off your own computer. Go back to [salamander](https://salamander.ai/) and click the 'Stop Server' button next to your server.

<img alt="" src="/images/salamander/stop.png" class="screenshot">

For more details, & updating the course / fastai library see "[Returning to work](update_salamander.html)".

## Advanced users: Connect via terminal

Click "Setup Access Key" and generate or upload an ssh key - it'll get added to all of your servers automatically. You can then copy & paste `ssh ubuntu@[xxx.xxx.xxx.xxx]` from the webpage to your terminal. Press enter and you're in!

Note: you should always generate keys yourself if you'd like to use them for several different platforms ([in-depth guide](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/))

---

_Many thanks to Ashton Six for writing this guide._
