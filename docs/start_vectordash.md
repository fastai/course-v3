---
title: Vectordash
sidebar: home_sidebar



---

This guide explains how to set up Vectordash to use PyTorch 1.0.0 and fastai 1.0.6. At the end of this
tutorial you will be able to use both in a GPU-enabled Jupyter Notebook environment.

## Pricing

The current cheapest Vectordash instance comes along with an Nvidia 1060 6GB GPU and costs $0.25 per
hour. You can try the more powerful GPU instances for
a few more cents per hour.

## Getting Set Up

### Creating your account

Vectordash offers cloud computing to fast.ai students. Cloud computing allows users access to virtual CPU or GPU resources on an hourly rate, depending on
the instance type. In case you do not already have an account on Vectordash, here is how to you can set
it up in less than 5 minutes:

1. Go to the [register page](http://vectordash.com/register).
2. Enter a valid email address and password.
3. Verify your email address with the confirmation link sent to your email.
4. Add a valid payment method (credit/debit card) to your account [here](http://vectordash.com/edit/payments).

Congrats! Your account is now setup.

### Start an instance

1. Go to the [create page](http://vectordash.com/create) to start an instance.
2. Under One-click Images, select the fast.ai image
3. Select the GPU type you want. *Please note the prices that correspond to each type.*
4. Enter a hostname
5. Click 'Create Instance'

<img alt="create_page" src="/home/chewing/course-v3/docs/images/vectordash/create_page.png" class="screenshot">

You will be redirected to the instance page. Leave it open since you will need some of the
information to setup the Vectordash command line interface (CLI) in the next step.

### Connect to your instance

We highly recommend using the vectordash-cli to interact with your Vectordash instance. With the
vectordash-cli, you can ssh into your instance, push/pull files, start a jupyter notebook with
one simple command, and more. To install and set it up, follow these steps:

1. Install pip:
   - MacOS/OS X: `sudo easy-install pip`
   - Linux: `sudo apt install python-pip`
2. `pip install vectordash -U`
3. `vectordash login`
   - Email: Vectordash account email
   - Secret: Vectordash secret token (can be found [here](http://vectordash.com/edit/verification))
4. `vectordash list`
5. `vectordash ssh $INSTANCE_ID`

<img alt="vectordash_cli" src="/home/chewing/course-v3/docs/images/vectordash/vectordash_cli.png" class="screenshot">

### Access fast.ai materials

After you have SSH-ed into to your instance, simply run the following command to access the
fast.ai materials:

`git clone https://github.com/fastai/course-v3`

If you would like to start a jupyter notebook on your instance, simply run the following command
on your local terminal:

`vectordash jupyter $INSTANCE_ID`

<img alt="jupyter" src="/home/chewing/course-v3/docs/images/vectordash/jupyter.png" class="screenshot">

If you have any problem while using the fastai library try running `conda update -all`. If you want
to read the vectordash-cli documentation, you can find it [here](http://vectordash.com/docs/cli).

### Stop an instance

You will be charged if you don't stop the instance while it's 'idle' (e.g. not training a network).
To stop an instance on Vectordash, go to the [dashboard](http://vectordash.com/dashboard) and click the
instance you would like to stop. Once on the instance page, click 'Stop Instance'. *Please note, stopping
an instance destroys it completely so make sure you save your files locally or in a remote storage location.*

<img alt="stop_instance" src="/home/chewing/course-v3/docs/images/vectordash/stop_instance.png" class="screenshot">