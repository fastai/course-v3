---

title: Salamander
sidebar: home_sidebar

---

# Welcome to Salamander!

It takes about 1 minute to signup & launch a Salamander server. The servers include everything you need to complete the fastai v3 course. Once launched, you can jump straight to Jupyter Notebook or connect directly via ssh.

## Pricing

Salamander tracks the AWS _spot_ price +26%. Prices at time of writing:

- K80: $0.36 per hour
- V100: $1.32 per hour

## Step 1: Create an account

Visit [https://salamander.ai](https://salamander.ai), click "Get Started", fill-in the form & add your card details.

![](/images/salamander/create_account.png)

## Step 2: Create your server

> If you already have a Salamander account, we recommend creating a brand new server to get the latest version of fastai

Pick your desired hardware & storage size (if you don't know what to choose, just keep the default options). Don't forget to accept the 'cuDNN Software License Agreement' and to check the four boxes on top of the button "Launch Server" before clicking it.

![](/images/salamander/create_server.png)

Wait about a minute for the server to start. You'll see the status update several times (written in orange) until it's ready like this:

![](/images/salamander/ready.png)

## Step 3: Connect to your Server

#### via Terminal

Click "Setup Access Key" and generate or upload an ssh key - it'll get added to all of your servers automatically. You can then copy & paste `ssh ubuntu@[xxx.xxx.xxx.xxx]` from the webpage to your terminal. Press enter and you're in! Note that if you choose to have Salamander generate a key for you, you shouldn't use it for any other servers.

#### Jupyter Notebook

Click 'Jupyter Notebook' to access the course materials. Once Jupyter Notebook loads, open "fastai-courses/course-v3/nbs/dl1/00_notebook_tutorial.ipynb".

When you're all done, **don't forget to shut down your instance**, so you don't get charged for all the time it's running in the background. It's not enough to just close your browser or turn off your own computer. Go back to the [salamander page](https://salamander.ai/) and click on the 'Stop Server' button next to your instance.

![](/images/salamander/stop.png)

## Step 4: Update fastai

Everything will be up-to-date once you've created your server. But for new changes you'll need to connect to your server & run this:

```sh
source activate fastai
conda update fastai
cd fastai-courses/course-v3
git pull
```

---

_Many thanks to Ashton Six for writing this guide._
