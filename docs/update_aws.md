---
title: Returning to AWS
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

Log in to the [AWS console](https://aws.amazon.com/console/) then click on the EC2 link (it should be in your history, otherwise find it in the 'Services' on the left or type EC2 in the search bar). Once on this page, either click on 'Instances' in the left menu or on the 'Running Instances' link.

<img alt="" src="/images/aws/instance.png" class="screenshot">

Tick the box of the instance you want to start, then click on 'Actions', scroll to 'Instance state' then click on 'Start'.

<img alt="" src="/images/aws/start.png" class="screenshot">

Note that in the 'Instance Settings' you can change your 'Instance Type'  while the instance is stopped. This can be extremely useful when you want to start on a lower-end machine type, test everything is okay, then move to a more powerful GPU.

You will have to wait a little bit for your instance to be ready while the light under instance state is orange.

<img alt="pending" src="/images/aws/pending.png" class="screenshot">

When it turns green, copy your instance IP in the IPv4 column.

<img alt="pubdns" src="/images/aws/pubdns.png" class="screenshot">

Open your terminal and use the command below (with IP_ADDRESS replaced by the ip address of your instance)

```
ssh -L8888:localhost:8888 ubuntu@IP_ADDRESS
```

If you want to update the course repository or the library (see below) you should do so now, and once you're ready type

```
jupyter notebook
```
You can then access your notebooks at [localhost:8888](http://localhost:8888).

### Update the course repo
 To update the course repo, while you're in the terminal, run those two instructions:

``` bash
cd course-v3
git pull
```

<img alt="" src="/images/gradient/update.png" class="screenshot">

This should give you the latest of the course notebooks. If you modified some of the notebooks in course-v3/nbs directly, GitHub will probably throw you an error. You should type `git stash` to remove your local changes. Remember you should always work on a copy of the lesson notebooks.

### Update the fastai library
To update the fastai library, while you're in the terminal  type
``` bash
conda update conda
conda install -c fastai fastai
```

### Stop your instance
When you finish working you must go back to your [AWS console](https://us-west-2.console.aws.amazon.com/ec2) and stop your instance manually to avoid getting extra charges. A good practice is setting a reminder for yourself (when you close your computer or log off) so you never forget to do it! 

Once in your EC2 console, click 'Instances' on the left menu then tick the box near your instance. Click on 'Actions', scroll down to 'Instance State' then choose 'Stop'.

<img alt="stop" src="/images/aws/stop.png" class="screenshot">

 **NOTE: you *will* be charged for the time that your instance is running.**

