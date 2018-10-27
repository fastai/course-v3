---
title: Returning to Sagemaker
keywords: 
sidebar: home_sidebar
---

To return to your notebook, the basic steps will be:

1. Start your instance
1. When done, shut down your instance

It is assumed you have started your SageMaker notebook instance following the steps outlined [here](start_sagemaker.html). This is necessary as your SageMaker notebook needs to have a lifecycle configuration attached. The lifecycle configuration has scripts that are run when the notebook instance is started to update the fastai library and also course content to ensure everything is up to date.

## Step by step guide

### Start your instance

Log in to the [AWS console](https://aws.amazon.com/console/) then click on the Sagemaker link (it should be in your history, otherwise find it in the 'Services' on the left or type sagemaker in the search bar). Once on this page, select 'Notebook instances' on the left menu.

<img alt="" src="/images/sagemaker/notebooks.png" class="screenshot">

Tick the box of the notebook you want to start, then click on 'Start'.

<img alt="" src="/images/sagemaker/start.png" class="screenshot">


You will have to wait a little bit for your instance to be ready while the light under instance state is orange. It will run the OnStart lifecycle configuration script to update both the course content and install the latest version of the fast.ai library.

<img alt="pending" src="/images/sagemaker/16.png" class="screenshot">

When it turns green, just click on 'Open' and you'll be back to your notebooks.

<img alt="ready" src="/images/sagemaker/17.png" class="screenshot">


### Stop your instance
When you finish working you must go back to your [AWS console](https://us-west-2.console.aws.amazon.com/sagemaker) and stop your instance manually to avoid getting extra charges. Just pick the notebook you want to stop and click on the 'Stop' button next to its name.

<img alt="stop" src="/images/sagemaker/23.png" class="screenshot">

 **NOTE: you *will* be charged for the time that your instance is running.**

