---
title: SageMaker
keywords: 
sidebar: home_sidebar
---

This is a quick guide to starting v3 of the fast.ai course Practical Deep Learning for Coders using AWS SageMaker. **NB: There is a temporary issue where data downloaded for training models, and saved models, are not saved after you shut down your instance. This will be resolved in a couple of weeks.**

## Pricing

The instance we suggest, p2.xlarge, is $1.26 an hour. The hourly rate is dependent on the instance type selected, see all available types [here](https://aws.amazon.com/sagemaker/pricing/).  Instances must be stopped to end billing.

## Getting Set Up

### Accessing SageMaker

1. Visit the [AWS webpage](https://aws.amazon.com/) and click on 'Sign In to the Console'. Next, enter your credentials if you are signing in or e-mail, account name and password if you need to sign up.

    <img src=/images/dlami_tutorial/signin.png" class="screenshot">

    If you do not have an account, the button to press will say 'Sign up' instead of 'Sign in to the Console'. If you are signing up you will also need to set your credit card details. This will be the credit card to which all the charges of the instance usage will be applied (if you have free credits you will not be charged until they are over). Note that you will also need to provide a phone number that will be called to verify your identity.

1. Once you have an account and are logged in, click *Services* in the top bar, and type 'sagemaker'. You can then click *Amazon SageMaker*.

    <img src=/images/sagemaker/01.png" class="screenshot">

### Configuring your notebook instance

1. On the left navigation bar, choose *Lifecycle Configurations*. This is where we set up the script that will create your notebook instance for you, with all software and lessons preinstalled.

    <img src=/images/sagemaker/03.png" class="screenshot">

1. Click *Create configuration*.

    <img src=/images/sagemaker/04.png" class="screenshot">

1. Enter *fastai* as the name.

    <img src=/images/sagemaker/05.png" class="screenshot">

1. In the *Scripts* section, click *Create notebook*. **NB:** ensure you are in the *Create notebook* section, otherwise your instance will be reconfigured from scratch every time you start it!

    <img src=/images/sagemaker/06.png" class="screenshot">

1. Paste the following to replace the script shown:

    ```bash
    #!/bin/bash
    set -e
    wget https://course-v3.fast.ai/setup/sagemaker
    chown ec2-user sagemaker
    sudo -H -u ec2-user -i bash -c 'nohup sh sagemaker &'
```

    <img src=/images/sagemaker/07.png" class="screenshot">

1. Click *Create configuration*..

    <img src=/images/sagemaker/08.png" class="screenshot">

1. On the left navigation bar, choose *Notebook instances*. This is where we create, manage, and access our notebook instances.

    <img src=/images/sagemaker/08b.png" class="screenshot">

1. Click *Create notebook instance*.

    <img src=/images/sagemaker/09.png" class="screenshot">

1. Enter *fastai* in the name, and in the instance type field choose *p2.xlarge*.

    <img src=/images/sagemaker/10.png" class="screenshot">

1. In the *IAM Role* section, choose to create a new role, then select *None* for S3 buckets, and choose *Create role*.

    <img src=/images/sagemaker/11.png" class="screenshot">

1. In the *Lifecycle configuration* section, choose the *fastai* configuration you created earlier.

    <img src=/images/sagemaker/12.png" class="screenshot">

1. Check that your selections now look like this:

    <img src=/images/sagemaker/13.png" class="screenshot">

1. Once it's entered correctly, click *Create notebook instance* at the bottom of the screen.

    <img src=/images/sagemaker/14.png" class="screenshot">

1. You will receive a message that the instance is being created.

    <img src=/images/sagemaker/15.png" class="screenshot">

1. For around 5 minutes it will show as *Pending* and you will not be able to access it.

    <img src=/images/sagemaker/16.png" class="screenshot">

### Accessing the notebooks

1. After about 5 minutes it will show *InService* and you can click *Open*.

    <img src=/images/sagemaker/17.png" class="screenshot">

1. Your server is now downloading and installing software in the background. You won't be able to see the course notebooks yet. Go get a cup of tea, and come back in 15 minutes.

    <img src=/images/sagemaker/18.png" class="screenshot">

1. After 15 minutes you should see a new *course-v3* folder has appeared, amongst others.

    <img src=/images/sagemaker/19.png" class="screenshot">

1. Click on that *course-v3* folder, then *nbs*, then *dl1*, and finally on *00_notebook_tutorial.ipynb*.

    <img src=/images/sagemaker/20.png" class="screenshot">

1. The first time you access this you'll get the following error - simply click the blue *Set Kernel* button to continue.

    <img src=/images/sagemaker/21.png" class="screenshot">

1. You're now ready to start using your first notebook! Simply follow the instructions to use the notebook.

    <img src=/images/sagemaker/22.png" class="screenshot">

### Shutting down your instance

- When you're done, close the notebook tab, and **remember to click stop!** If you don't, you'll keep getting charged until you click the *stop* button.

    <img src=/images/sagemaker/23.png" class="screenshot">

## More help

For questions or issues related to course content, we recommend posting in the [fast.ai forum](http://forums.fast.ai/).

