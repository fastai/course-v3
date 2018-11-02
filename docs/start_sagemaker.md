---
title: SageMaker
keywords: 
sidebar: home_sidebar
---

This is a quick guide to starting v3 of the fast.ai course Practical Deep Learning for Coders using AWS SageMaker. 

If you are returning to work and have previously completed the steps below, please go to the [returning to work](http://course-v3.fast.ai/update_sagemaker.html) section.

## Pricing

The instance we suggest, ml.p2.xlarge, is $1.26 an hour. The hourly rate is dependent on the instance type selected, see all available types [here](https://aws.amazon.com/sagemaker/pricing/).  You will need to explicitely request a limit request to use this instance, [here](https://course-v3.fast.ai/start_aws.html#step-2-request-service-limit ) Instances must be stopped to end billing.

## Getting Set Up

### Accessing SageMaker

1. Visit the [AWS webpage](https://aws.amazon.com/) and click on 'Sign In to the Console'. Next, enter your credentials if you are signing in or e-mail, account name and password if you need to sign up.

    <img alt="stop" src="/images/aws/signin.png" class="screenshot">

    If you do not have an account, the button to press will say 'Sign up' instead of 'Sign in to the Console'. If you are signing up you will also need to set your credit card details. This will be the credit card to which all the charges of the instance usage will be applied (if you have free credits you will not be charged until they are over). Note that you will also need to provide a phone number that will be called to verify your identity.

1. Once you have an account and are logged in, click *Services* in the top bar, and type 'sagemaker'. You can then click *Amazon SageMaker*.

   <img alt="sage" src="/images/sagemaker/01.png" class="screenshot">

### Configuring your notebook instance

1. On the left navigation bar, choose *Lifecycle Configurations*. This is where we set up the script that will create your notebook instance for you, with all software and lessons preinstalled.

    <img alt="lifecycle config" src="/images/sagemaker/03.png" class="screenshot">

1. Click *Create configuration*.

   <img alt="create config" src="/images/sagemaker/04.png" class="screenshot">

1. Enter *fastai* as the name.

    <img alt="fastai" src="/images/sagemaker/05.png" class="screenshot">

1. In the *Scripts* section, click *Start notebook*. 

    <img alt="create" src="/images/sagemaker/06.png" class="screenshot">

1. Paste the following to replace the script shown:

    ```bash
    #!/bin/bash
    wget -N https://course-v3.fast.ai/setup/sagemaker-start;
    chown ec2-user sagemaker-start;
    chmod u+x sagemaker-start;
    sudo -H -u ec2-user -i bash -c './sagemaker-start';
    ```

1. In the *Scripts* section, click *Create notebook*. **NB:** ensure you are in the *Create notebook* section, otherwise your instance will be reconfigured from scratch every time you start it!

    <img alt="create" src="/images/sagemaker/06.png" class="screenshot">

1. Paste the following to replace the script shown:

    ```bash
    #!/bin/bash
    wget -N https://course-v3.fast.ai/setup/sagemaker-create;
    chown ec2-user sagemaker-create;
    chmod u+x sagemaker-create;
    sudo -H -u ec2-user -i bash -c 'nohup ./sagemaker-create &';
    ```

    <img alt="script" src="/images/sagemaker/07.png" class="screenshot">

1. Click *Create configuration*..

    <img alt="create" src="/images/sagemaker/08.png" class="screenshot">

1. On the left navigation bar, choose *Notebook instances*. This is where we create, manage, and access our notebook instances.

    <img alt="notebook instance" src="/images/sagemaker/08b.png" class="screenshot">

1. Click *Create notebook instance*.

    <img alt="create nb instance" src="/images/sagemaker/09.png" class="screenshot">

1. Enter *fastai* in the name, and in the instance type field choose *ml.p2.xlarge*.

    <img alt="choose ml.p2" src="/images/sagemaker/10.png" class="screenshot">

1. In the *IAM Role* section, choose to create a new role, then select *None* for S3 buckets, and choose *Create role*.

   <img alt="role" src="/images/sagemaker/11.png" class="screenshot">

1. In the *Lifecycle configuration* section, choose the *fastai* configuration you created earlier.

    <img alt="config" src="/images/sagemaker/12.png" class="screenshot">

1. In the *Volume Size in GB - optional* section, enter a volume size between 15 and 25 GB (we recommend 25 GB).

    <img alt="config" src="/images/sagemaker/24.png" class="screenshot">

1. Check that your selections now look like this:

    <img alt="summary" src="/images/sagemaker/13.png" class="screenshot">

1. Once it's entered correctly, click *Create notebook instance* at the bottom of the screen.

    <img alt="click" src="/images/sagemaker/14.png" class="screenshot">

1. You will receive a message that the instance is being created.

    <img alt="message" src="/images/sagemaker/15.png" class="screenshot">

1. For around 5 minutes it will show as *Pending* and you will not be able to access it.

   <img alt="pending" src="/images/sagemaker/16.png" class="screenshot">

### Accessing the notebooks

1. After about 5 minutes it will show *InService* and you can click *Open*.

    <img alt="in service" src="/images/sagemaker/17.png" class="screenshot">

1. Your server is now downloading and installing software in the background. You won't be able to see the course notebooks yet. Go get a cup of tea, and come back in 15 minutes.

    <img alt="server" src="/images/sagemaker/18.png" class="screenshot">

1. After 15 minutes you should see a new *course-v3* folder has appeared, amongst others.

    <img alt="course" src="/images/sagemaker/19.png" class="screenshot">

1. When you start the notebook, if prompted (not expected if all is well) to select a kernel choose *Python 3*. If you aren't prompted, you can verify the kernel name on the top right hand side, you can change the attahed kernel through the menu *Kernel > Change Kernel*

1. Go back to the [first page](index.html) to see how to use this jupyter notebook and run the jupyter notebook tutorial. Come back here once you're finished and *don't forget to stop your instance* with the next step.

### Shutting down your instance

- When you're done, close the notebook tab, and **remember to click stop!** If you don't, you'll keep getting charged until you click the *stop* button.

    <img alt="stop" src="/images/sagemaker/23.png" class="screenshot">

  To see how to open it again, update the course or the fastai library, go to the [Returning to work page](update_sagemaker.html).

## More help

For questions or issues related to course content, we recommend posting in the [fast.ai forum](http://forums.fast.ai/).

