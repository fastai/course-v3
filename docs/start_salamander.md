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

Pick your desired hardware & storage size (if you don't know what to choose, just keep the default options). Don't forget to accept the 'cuDNN Software License Agreement' and to check the acknowledgements above "Launch Server" before clicking it.

![](/images/salamander/create_server.png)

Wait about a minute for the server to start. Once finished, it will look like this:

![](/images/salamander/ready.png)

## Step 3: Open Jupyter Notebook

Click 'Jupyter Notebook' to access the course materials. After Jupyter Notebook loads, click on 'fastai_courses' & then 'course-v3'.

![nb tuto](/images/salamander/final.png)

See [here](index#jupyter-notebook) for instructions on running the Jupyter Notebook tutorial. Return to this guide once you're finished and _don't forget to stop your server_.

## Step 4: Stop your server

When you're all done, **don't forget to shut down your server**, so you don't get charged for the time it's running in the background. It's not enough to just close your browser or turn off your own computer. Go back to [salamander](https://salamander.ai/) and click the 'Stop Server' button next to your server.

![](/images/salamander/stop.png)

For more details, & updating the course / fastai library see "[Returning to work](update_salamander)".

## Advanced users: Connect via terminal

Click "Setup Access Key" and generate or upload an ssh key - it'll get added to all of your servers automatically. You can then copy & paste `ssh ubuntu@[xxx.xxx.xxx.xxx]` from the webpage to your terminal. Press enter and you're in!

Note: you should always generate keys yourself if you'd like to use them for several different platforms ([in-depth guide](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/))

---

_Many thanks to Ashton Six for writing this guide._
