---
title: SageMaker
keywords: 
sidebar: home_sidebar
---

This is a quick guide to starting v3 of the fast.ai course Practical Deep Learning for Coders using Amazon SageMaker. 

If you are returning to work and have previously completed the steps below, please go to the [returning to work](https://course.fast.ai/update_sagemaker.html) section.

We will use [AWS CloudFormation](https://aws.amazon.com/cloudformation/) to provision all of the SageMaker resources including the Notebook instance, Notebook Lifecyle configuration and IAM role. By default it will provision a SageMaker notebook instance of type *ml.p2.xlarge* which has the Nvidia K80 GPU and 50 GB of EBS disk space.

## Pricing

The instance we suggest, ml.p2.xlarge, is $1.26 an hour. The hourly rate is dependent on the instance type selected, see all available types [here](https://aws.amazon.com/sagemaker/pricing/).  You will need to explicitely request a limit request to use this instance or the ml.p3.2xlarge instance, [here](https://course.fast.ai/start_aws.html#step-2-request-service-limit ) Instances must be stopped to end billing.

## Getting Set Up

### Creating the SageMaker Notebook Instance

1. Visit the [AWS webpage](https://aws.amazon.com/) and click on 'Sign In to the Console'. Next, enter your credentials if you are signing in or e-mail, account name and password if you need to sign up.

    <img alt="signin" src="/images/aws/signin.png" class="screenshot">

    If you do not have an account, the button to press will say 'Sign up' instead of 'Sign in to the Console'. If you are signing up you will also need to set your credit card details. This will be the credit card to which all the charges of the instance usage will be applied (if you have free credits you will not be charged until they are over). Note that you will also need to provide a phone number that will be called to verify your identity.

1. Once you have an account and are logged in we are ready to create all the SageMaker resources using CloudFormation. To lauch the CloudFormation stack click the *Launch Stack* button for the closest region to where you live **in the table below** . 

    Region | Name | Launch link
    --- | --- | ---
    US West (Oregon) Region | us-west-2 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)
    US East (N. Virginia) Region | us-east-1 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://us-east-1.console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)
    US East (Ohio) Region | us-east-2 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://us-east-2.console.aws.amazon.com/cloudformation/home?region=us-east-2#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)
    US East (N. California) Region | us-west-1 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://us-west-1.console.aws.amazon.com/cloudformation/home?region=us-west-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)    
    Asia Pacific (Tokyo) Region | ap-northeast-1 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://ap-northeast-1.console.aws.amazon.com/cloudformation/home?region=ap-northeast-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)
    Asia Pacific (Seoul) Region | ap-northeast-2 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://ap-northeast-2.console.aws.amazon.com/cloudformation/home?region=ap-northeast-2#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)
    Asia Pacific (Sydney) Region | ap-southeast-2 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://ap-southeast-2.console.aws.amazon.com/cloudformation/home?region=ap-southeast-2#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)
    Asia Pacific (Mumbai) Region | ap-south-1 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://ap-south-1.console.aws.amazon.com/cloudformation/home?region=ap-south-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack) 
    Asia Pacific (Singapore) Region | ap-southeast-1 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://ap-southeast-1.console.aws.amazon.com/cloudformation/home?region=ap-southeast-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)           
    Canada (central) Region | ca-central-1 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://ca-central-1.console.aws.amazon.com/cloudformation/home?region=ca-central-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)       
    EU (Ireland) Region | eu-west-1 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://eu-west-1.console.aws.amazon.com/cloudformation/home?region=eu-west-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)
    EU (Frankfurt) Region | eu-central-1 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://eu-central-1.console.aws.amazon.com/cloudformation/home?region=eu-central-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)
    EU (London) Region | eu-west-2 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://eu-west-2.console.aws.amazon.com/cloudformation/home?region=eu-west-2#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fsagemaker-cfn.yml&stackName=FastaiSageMakerStack)    

1. This will open the AWS CloudFormation web console with the template to create the AWS resources as per the screenshot below. Take a look at the input parameters and either accept the defaults or update them as necessary. Select the option box **I acknowledge that AWS CloudFormation might create IAM resources.** and then click the **Create** button to create the stack.

    <img alt="create stack" src="/images/sagemaker/create_stack.png" class="screenshot">

1. You will see the following CloudFormation page showing the stack is being created. Once the stack reaches the **CREATE_COMPLETE** state then open the AWS web console and click *Services* in the top bar, and type 'sagemaker'. You can then click *Amazon SageMaker*.

   <img alt="sage" src="/images/sagemaker/01.png" class="screenshot">

1. On the left navigation bar, choose *Notebook instances*. This is where we create, manage, and access our notebook instances. You should see that your notebook instance named **fastai** status has the status *InService* as per the screenshot below.

   <img alt="pending" src="/images/sagemaker/17.png" class="screenshot">

1. Click on the *Open Jupyter* link to open your Jupyter web console.

### Shutting down your instance

- When you're done, close the notebook tab, and **remember to click stop!** If you don't, you'll keep getting charged until you click the *stop* button.

    <img alt="stop" src="/images/sagemaker/23.png" class="screenshot">

  To see how to open it again, update the course or the fastai library, go to the [Returning to work page](update_sagemaker.html).

## Troubleshooting installation problems

- If you do not receive a notifcation email after more than 15 minutes then there may have been a problem installing the fast.ai libraries and dependencies on your notebook instance. To troubleshoot, open the [AWS console](https://aws.amazon.com/console/) then click on the **CloudWatch** link (type *cloudwatch* in the search bar). Once you are in the CloudWatch console, navigate to *Logs -> /aws/sagemaker/NotebookInstances -> fastai/LifecycleConfigOnStart* or *fastai/LifecycleConfigOnCreate* to view the output of the installation scripts.

## More help

For questions or issues related to course content, we recommend posting in the [fast.ai forum](http://forums.fast.ai/).
