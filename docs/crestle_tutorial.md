---

title: Crestle.ai
keywords: neuralnets deeplearning
sidebar: home_sidebar

---
# Welcome to Crestle.ai!
![](./images/crestle/landing_page.png)

[Crestle.ai](https://www.crestle.ai/) is an effortless infrastructure for deep learning. Once you sign up you should be able to spin up a GPU enabled Jupyter notebook within a minute. There is no setup required on your part.

## Using crestle.ai for fast.ai v3 course

crestle.ai comes bundled with fast.ai course setup, including all the datasets (~35 GB) required as part of the course. You can just dive right into the relevant notebook and start training!

![](./images/crestle/jupyter_fast_ai_repo.png)

## Pricing

Crestle uses AWS [p2.xlarge _spot_](https://aws.amazon.com/ec2/instance-types/p2/) instances to provision a GPU instance. Using spot instance allows us to keep the costs low. We are hosted on multiple AWS regions thus providing a better guarantee on availability of instances.
Every GPU-enabled notebook is backed by a dedicated [NVIDIA Tesla K80 GPU](https://www.nvidia.com/en-us/data-center/tesla-k80/).

Our goal is to keep the costs as low as possible for students and fast.ai practitioners. At the time of this writing we are charging:

- K80 GPU instance: $0.30 per hour, billed per second
- Storage: $0 (upto 75 GB) until 12/31/2018, after which it will be $0.10/GB/month.

You also get 1 hour of free GPU usage when you sign up.

![](./images/crestle/pricing.png)

## Getting started

### Step 1: Create a crestle.ai account
Sign up for Crestle [here](https://www.crestle.ai/). Verify your email by clicking on the link Crestle emails you. Once you have verified, you can sing in to your account.

![](./images/crestle/signup.png)

### Step 2: Start your Jupyter notebook
Once you login you can start your Jupyter notebook with a press of a button. You will not be billed until you start your Jupyter notebook.

![](./images/crestle/start_jupyter.png)

### Step 3: Navigate to fast.ai course
Once you start the notebook your GPU instance should be ready within a minute. We have already setup fast.ai for you. The directory structure available for you out of the box is as below:

```
- courses
     - fast-ai
        - course-v3
```

![](./images/crestle/jupyter_fast_ai_repo.png)

### Step 4: The datasets are already there - start coding!

We have already mounted the [datasets](http://course.fast.ai/datasets) required for this course. You don't have to download them. This is about 35 GB worth of data ready for you to train your models on.

![](./images/crestle/datasets.png)

You can navigate into the relevant notebook and start building.

![](./images/crestle/lesson1.png)

### Step 5: Stop the instance when you are done

You need to **stop the instance** when you are done. This ensures you are not charged for time you are not spending training or coding. The stop button is located the top. You will be able to start your Jupyter instance as described in Step 2 any time.

Shutting down your instance will backup and snapshot the data from your home directory and will be avaible for you the next time you launch the instance.

![](./images/crestle/stop_jupyter.png)

## What is installed?

crestle.ai has provisioned a [full Anaconda 5.3.0 environment](https://repo.anaconda.com/archive/) which bundles all the popular scientific packages.

Additonally we have also setup the fast.ai environment as described [here](https://github.com/fastai/fastai#conda-install) which includes Pytorch with GPU support and fastai library as well.

You can open the Terminal from Jupyter notebook and install additional libraries using `conda install` or even `pip install` which will install this on your home directory and will be backed up for you when you shutdown your instance.


## Storage and Data

With crestle.ai we provide up to 75GB worth of storage for you. Please note that about 37GB of this is already taken for loading the datasets, anaconda and its dependencies.

You have complete read/write access to all this. Your home directory /home/nbuser contains the courses directory and anaconda install directory (hidden directory).

Anything you add to your home directory will be backed up for you for free.


## FAQs

### How is this different from crestle.com?

crestle.ai is a new version of crestle.com which was previously setup for fast.ai course. With crestle.ai we have optimized the setup to run the latest fast.ai v3 course. We have also made it much faster by moving away from EFS setup and have significantly reduced the price!


### Where can I get support?

Just send an email to support@crestle.com - we are happy to help.


### Who is behind Crestle?

crestle.com was created by [Anurag Goel](https://twitter.com/anuraggoel) and is now being supported and managed by [doc.ai](https://doc.ai/).

### Where can I learn more about Crestle?

Please read our [FAQ](https://www.crestle.ai/faq). Our pricing information is listed [here](https://www.crestle.ai/pricing). You can always reach out to us at support@crestle.com.