---
title: SageMaker
keywords: 
sidebar: home_sidebar
---

This is a quick guide to starting v3 of the fast.ai course Practical Deep Learning for Coders using AWS SageMaker. 

If you are returning to work and have previously completed the steps below, please go to the [returning to work](http://course-v3.fast.ai/update_sagemaker.html) section.

We will use [AWS CloudFormation](https://aws.amazon.com/cloudformation/) to provision the AWS resources needed including the SageMaker notebook instance and associated Notebook Lifecycle Configs, IAM role and an SNS topic. The SNS topic is used to alert you when the SageMaker Notebook has all the necessary fast.ai libraries installed and is ready for use.

**NB: There is a temporary issue where data downloaded for training models, and saved models, are not saved after you shut down your instance. This will be resolved in a couple of weeks.**

## Pricing

The instance we suggest, ml.p2.xlarge, is $1.26 an hour. The hourly rate is dependent on the instance type selected, see all available types [here](https://aws.amazon.com/sagemaker/pricing/).  You will need to explicitely request a limit request to use this instance or the ml.p3.2xlarge instance, [here](https://course-v3.fast.ai/start_aws.html#step-2-request-service-limit ) Instances must be stopped to end billing.

## Getting Set Up

### Creating the AWS resources via CloudFormation

1. Visit the [AWS webpage](https://aws.amazon.com/) and click on 'Sign In to the Console'. Next, enter your credentials if you are signing in or e-mail, account name and password if you need to sign up.

    <img alt="signin" src="/images/aws/signin.png" class="screenshot">

    If you do not have an account, the button to press will say 'Sign up' instead of 'Sign in to the Console'. If you are signing up you will also need to set your credit card details. This will be the credit card to which all the charges of the instance usage will be applied (if you have free credits you will not be charged until they are over). Note that you will also need to provide a phone number that will be called to verify your identity.

1. Once you have an account and are logged in, click the *Launch Stack* button for the closest region to where you live in the table below. 

    Region | Name | Launch link
    --- | --- | ---
    US West (Oregon) Region | us-west-2 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fcfn.yml&stackName=FastaiSageMakerNbStack)
    US East (N. Virginia) Region | us-east-1 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://us-east-1.console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fcfn.yml&stackName=FastaiSageMakerNbStack)
    US East (Ohio) Region | us-east-2 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://us-east-2.console.aws.amazon.com/cloudformation/home?region=us-east-2#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fcfn.yml&stackName=FastaiSageMakerNbStack)
    Asia Pacific (Tokyo) Region | ap-northeast-1 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://ap-northeast-1.console.aws.amazon.com/cloudformation/home?region=ap-northeast-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fcfn.yml&stackName=FastaiSageMakerNbStack)
    Asia Pacific (Seoul) Region | ap-northeast-2 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://ap-northeast-2.console.aws.amazon.com/cloudformation/home?region=ap-northeast-2#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fcfn.yml&stackName=FastaiSageMakerNbStack)
    Asia Pacific (Sydney) Region | ap-southeast-2 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://ap-southeast-2.console.aws.amazon.com/cloudformation/home?region=ap-southeast-2#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fcfn.yml&stackName=FastaiSageMakerNbStack)
    EU (Ireland) Region | eu-west-1 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://eu-west-1.console.aws.amazon.com/cloudformation/home?region=eu-west-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fcfn.yml&stackName=FastaiSageMakerNbStack)
    EU (Frankfurt) Region | eu-central-1 | [![CloudFormation](/images/aws/cfn-launch-stack.png)](https://eu-central-1.console.aws.amazon.com/cloudformation/home?region=eu-central-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fsagemaker-fastai-notebook%2Fcfn.yml&stackName=FastaiSageMakerNbStack)

1. This will open the AWS CloudFormation web console with the template to create the AWS resources already loaded as per the screenshot below. You only need to enter 3 input parameters: one for the Notebook instance type (default value is *ml.p2.xlarge*), another for the fastai libaray version (default value is *1.0*) and finally your email address to receive an notification email when the fastai library is installed on your notebook instance and ready for use. Tick the option box to acknowledge that IAM resources will be created and then click the *Create* button to create the stack.

    <img alt="create_stack" src="/images/aws/cfn_create_stack.png" class="screenshot">

1. You will see the following CloudFormation page showing the stack is being created.

    <img alt="in_progress" src="/images/aws/cfn_stack_detail_in_progress.png" class="screenshot">

1. While your Cloudformation stack is creating the AWS resources, you will receive an SNS subscription confirmation email to the email address supplied in the input parameters to the CloudFormation stack similar to the screenshot below. Once you receive the email, click the *Confirm subscription* link. You need to do this to receive the notification email when the fast.ai library has been installed correctly. 

    <img alt="confirm_sub" src="/images/aws/confirm_sub.png" class="screenshot">

1. Wait for up to 15 minutes to have the fast.ai library and dependencies installed on the SageMaker notebook instance. When it has done you will receive an email with a *tinyurl.com* link to open the Jupyter console of the notebook instance as per the screenshot below. 

    <img alt="email" src="/images/aws/email_notification_ready.png" class="screenshot">

1. You should now have the Jupyter console opened with a screen similar to the one shown below.

    <img alt="jupyter" src="/images/aws/jupyter_nb.png" class="screenshot">

1. When you start the notebook, if prompted (not expected if all is well) to select a kernel choose *Python 3*. If you aren't prompted, you can verify the kernel name on the top right hand side, you can change the attahed kernel through the menu *Kernel > Change Kernel*

1. Go back to the [first page](index.html) to see how to use this jupyter notebook and run the jupyter notebook tutorial. Come back here once you're finished and *don't forget to stop your instance* with the next step.

### Shutting down your instance

- When you're done, close the notebook tab, and **remember to click stop!** If you don't, you'll keep getting charged until you click the *stop* button.

    <img alt="stop" src="/images/sagemaker/23.png" class="screenshot">

  To see how to open it again, update the course or the fastai library, go to the [Returning to work page](update_sagemaker.html).

### Troubleshooting installation problems

- If you do not receive a notifcation email after more than 15 minutes then there may have been a problem installing the fast.ai libraries and dependencies on your notebook instance. To troubleshoot, open the [AWS console](https://aws.amazon.com/console/) then click on the **CloudWatch** link (type *cloudwatch* in the search bar). Once you are in the CloudWatch console, navigate to *Logs -> /aws/sagemaker/NotebookInstances -> fastai/LifecycleConfigOnStart* or *fastai/LifecycleConfigOnCreate* to view the output of the installation scripts.

## More help

For questions or issues related to course content, we recommend posting in the [fast.ai forum](http://forums.fast.ai/).

