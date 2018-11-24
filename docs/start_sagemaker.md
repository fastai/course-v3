---
title: SageMaker
keywords: 
sidebar: home_sidebar
---

This is a quick guide to starting v3 of the fast.ai course Practical Deep Learning for Coders using AWS SageMaker. 

If you are returning to work and have previously completed the steps below, please go to the [returning to work](http://course-v3.fast.ai/update_sagemaker.html) section.

We will use [AWS CloudFormation](https://aws.amazon.com/cloudformation/) to provision the SageMaker notebook lifecycle configuration and IAM role for the notebook instance. We will create the SageMaker notebook instance manually.

## Pricing

The instance we suggest, ml.p2.xlarge, is $1.26 an hour. The hourly rate is dependent on the instance type selected, see all available types [here](https://aws.amazon.com/sagemaker/pricing/).  You will need to explicitely request a limit request to use this instance or the ml.p3.2xlarge instance, [here](https://course-v3.fast.ai/start_aws.html#step-2-request-service-limit ) Instances must be stopped to end billing.

## Getting Set Up

### Creating the SageMaker Notebook Lifecycle Config and IAM role via CloudFormation

1. Visit the [AWS webpage](https://aws.amazon.com/) and click on 'Sign In to the Console'. Next, enter your credentials if you are signing in or e-mail, account name and password if you need to sign up.

    <img alt="signin" src="/images/aws/signin.png" class="screenshot">

    If you do not have an account, the button to press will say 'Sign up' instead of 'Sign in to the Console'. If you are signing up you will also need to set your credit card details. This will be the credit card to which all the charges of the instance usage will be applied (if you have free credits you will not be charged until they are over). Note that you will also need to provide a phone number that will be called to verify your identity.

1. Once you have an account and are logged in we are ready to create the SageMaker Notebook Lifecycle Configuration and IAM role that will be both linked to the SageMaker Notebook Instance. Click the *Launch Stack* button for the closest region to where you live **in the table below** . 

    Region | Name | Launch link
    --- | --- | ---
    US West (Oregon) Region | us-west-2 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)
    US East (N. Virginia) Region | us-east-1 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://us-east-1.console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)
    US East (Ohio) Region | us-east-2 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://us-east-2.console.aws.amazon.com/cloudformation/home?region=us-east-2#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)
    Asia Pacific (Tokyo) Region | ap-northeast-1 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://ap-northeast-1.console.aws.amazon.com/cloudformation/home?region=ap-northeast-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)
    Asia Pacific (Seoul) Region | ap-northeast-2 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://ap-northeast-2.console.aws.amazon.com/cloudformation/home?region=ap-northeast-2#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)
    Asia Pacific (Sydney) Region | ap-southeast-2 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://ap-southeast-2.console.aws.amazon.com/cloudformation/home?region=ap-southeast-2#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)
    EU (Ireland) Region | eu-west-1 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://eu-west-1.console.aws.amazon.com/cloudformation/home?region=eu-west-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)
    EU (Frankfurt) Region | eu-central-1 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://eu-central-1.console.aws.amazon.com/cloudformation/home?region=eu-central-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)

1. This will open the AWS CloudFormation web console with the template to create the AWS resources already loaded as per the screenshot below. Tick the option box to acknowledge that IAM resources will be created and then click the *Create* button to create the stack.

    <img alt="create stack" src="/images/sagemaker/create_stack.png" class="screenshot">

1. You will see the following CloudFormation page showing the stack is being created. Take note of the Notebook Lifecycle Config resource and IAM role created by CloudFormation. You will need these values when manually creating the SageMaker Notebook instance.

    <img alt="stack complete" src="/images/sagemaker/stack_complete.png" class="screenshot">

1. Once the stack reaches the *CREATE_COMPLETE* state then open the AWS web console and click *Services* in the top bar, and type 'sagemaker'. You can then click *Amazon SageMaker*.

   <img alt="sage" src="/images/sagemaker/01.png" class="screenshot">

1. On the left navigation bar, choose *Notebook instances*. This is where we create, manage, and access our notebook instances.

    <img alt="notebook instance" src="/images/sagemaker/08b.png" class="screenshot">

1. Click *Create notebook instance*.

    <img alt="create nb instance" src="/images/sagemaker/09.png" class="screenshot">

1. Enter *fastai* in the name, and in the instance type field enter *ml.p2.xlarge* or *ml.p3.2xlarge* and for volume size enter *50* GB. For the IAM role and Lifecycle Config copy the output values from the CloudFormation stack output. It should look something similar to the screenshot below.

   <img alt="create notebook" src="/images/sagemaker/sagemaker_notebook_create.png" class="screenshot">

1. Once it's entered correctly, click *Create notebook instance* at the bottom of the screen.

    <img alt="click" src="/images/sagemaker/14.png" class="screenshot">

1. You will receive a message that the instance is being created.

    <img alt="message" src="/images/sagemaker/15.png" class="screenshot">

1. For around 5 minutes it will show as *Pending* and you will not be able to access it.

   <img alt="pending" src="/images/sagemaker/16.png" class="screenshot">

1. It will take around 10 minutes to fully setup your notebook instance with the fastai library. You will know when it is ready when the Jupyter kernel name *Python 3* is available and a 'course-v3' folder appears in your Jupyter Notebook window.

### Shutting down your instance

- When you're done, close the notebook tab, and **remember to click stop!** If you don't, you'll keep getting charged until you click the *stop* button.

    <img alt="stop" src="/images/sagemaker/23.png" class="screenshot">

  To see how to open it again, update the course or the fastai library, go to the [Returning to work page](update_sagemaker.html).

### Troubleshooting installation problems

- If you do not receive a notifcation email after more than 15 minutes then there may have been a problem installing the fast.ai libraries and dependencies on your notebook instance. To troubleshoot, open the [AWS console](https://aws.amazon.com/console/) then click on the **CloudWatch** link (type *cloudwatch* in the search bar). Once you are in the CloudWatch console, navigate to *Logs -> /aws/sagemaker/NotebookInstances -> fastai/LifecycleConfigOnStart* or *fastai/LifecycleConfigOnCreate* to view the output of the installation scripts.

## More help

For questions or issues related to course content, we recommend posting in the [fast.ai forum](http://forums.fast.ai/).
