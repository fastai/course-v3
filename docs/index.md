---
title: Practical Deep Learning for Coders, v3
---

## Getting started

Welcome! If you're new to all this deep learning stuff, then don't worry&mdash;we'll take you through it all step by step. We do however assume that you've been coding for at least a year, and also that (if you haven't used Python before) you'll be putting in the extra time to learn whatever Python you need as you go. (For learning Python, we've heard good things about the [DataCamp](https://www.datacamp.com/courses/intro-to-python-for-data-science) course, as well as [Learn Python the Hard Way](https://learnpythonthehardway.org/).)

You might be surprised by what you *don't* need to become a top deep learning practitioner. You need one year coding experience, a GPU and appropriate software (see below), and that's it. You don't need much data, you don't need university-level math, and you don't need a giant data center. For more on this, see our article: [What you need to do deep learning](http://www.fast.ai/2017/11/16/what-you-need/).

### Using a GPU

To do nearly everything in this course, you'll need access to a computer with an NVIDIA GPU (unfortunately other brands of GPU are not fully supported by the main deep learning libraries). However, we don't recommend you buy one; in fact, even if you already have one, we don't suggest you use it just yet! Setting up a computer takes time and energy, and you want all your energy to focus on deep learning right now. Plus, the fastai library uses the latest version of pytorch that doesn't support Windows or MacOS yet. Therefore, we instead suggest you rent access to a computer that already has everything you need preinstalled and ready to go. Costs will generally be around US$0.36 to US$1.26 per hour while you're using it. 

 The most important thing to remember: **when you're done, shut down your server**. You will be renting a distant computer, not running something on your own. It's not enough to close your browser, or turn off your own PC, those will merely sever the connection between your device and this distant server, not shut down the thing you're paying. You have to shut this server down using the methods described in the guides below. Otherwise, you'll be charged for all the time it runs and get surprised with a nasty bill!

Here are some great choices of platforms. Click the link for more information on each, and setup instructions.

#### Ready to run options

These are the easiest to use; they've got all the software, data, and lessons preinstalled for you.

- [Paperspace Gradient](/start_gradient.html); (instant approval, no installation required, $0.59 an hour + storage, $10 free credit)
- [Crestle](/start_crestle.html); (instant approval, no installation required, Free compute until Nov-15th-2018. After that $0.30 an hour for compute + storage costs from Jan-1-2019)
- [Salamander](/start_salamander.html); (instant approval, no installation required, $0.36 an hour + storage, $1 free credit)
- [Colab](/start_colab.html); (instant approval, requires minimal installation, free)
- [SageMaker](/start_sagemaker.html); (requires some installation, and wait for approval, $1.26 an hour + storage)
- [Floydhub](/start_floydhub.html); (instant approval, no installation required, $1.20/hour + $9.00/month (100GB storage), 2 hours free credit)

#### Some installation required

If you're comfortable at a command line, these options are fairly easy to get started with, and may be more flexible in the long term than the options above.

- [Google Compute Platform](/start_gcp.html) ($0.38 an hour + storage, $300 free credit)
- [Amazon Web Services EC2](/start_aws.html) ($0.9 an hour + storage)

**For those starting out, we highly recommend a Jupyter Notebooks platform (Option 1)**

* Notebooks are the easiest way to start writing python code and experimenting with deep learning.  
* Renting a Cloud Server (Option 2) requires environment configuration and setup.  
* Building a PC requires environment setup and more up-front money. 

(When we release Part 2 of the course, we will go into more specific details and benefits on both building a PC and renting a server.)

### Jupyter notebook

Once you've finished the steps in one of the guide above, you'll be presented with a screen like this.

<img alt="" src="/images/jupyter.png" class="screenshot">

 This is the jupyter notebook environment, where you'll be doing nearly all your work in the course, so you'll want to get very familiar with it! You'll be learning a bit about it during the course, but you should probably spend a moment to try out the notebook tutorial.

Your first task, then, is to open this notebook tutorial! To do so, click `nbs` and then `dl1` in jupyter, where you'll then see all the lesson notebooks. First, tick the little box on the left of `00_notebook_tutorial.ipynb` then click duplicate. 

<img alt="" src="/images/duplicate.png" class="screenshot">

You want to avoid modifying the original course notebooks as you will get conflicts when you try to update this folder with GitHub (the place where the course is hosted). But we also want you to try a lot of variations of what is shown in class, which is why we encourage you to use duplicates of the course notebooks. 

Launch you copy of `00_notebook_tutorial.ipynb` and follow the instructions!

When you're done, **remember to shut down your server**.

### Our forums

Got stuck? Want to know more about some topic? Your first port of call should be [forums.fast.ai](https://forums.fast.ai/). There are thousands of students and practitioners asking and answering questions there. That means that it's likely your question has already been answered! So click the little magnifying class in the top right there, and search for the information you need; for instance, if you have some error message, paste a bit of it into the search box.

The forum software we use is called [Discourse](https://www.discourse.org/about). When you first join, it will show you some tips and tricks. There is also this [handy walk-thru](https://forums.episodeinteractive.com/t/a-quick-how-to-for-discourse/48/1) provided by another Discourse forum (not affiliated with fast.ai).

### PyTorch and fastai

We teach how to train [PyTorch](https://pytorch.org/) models using the [fastai](https://docs.fast.ai) library. These two pieces of software are deeply connected&mdash;you can't become really proficient at using fastai if you don't know PyTorch well too. Therefore, you will often need to refer to the [PyTorch docs](https://pytorch.org/docs/stable/index.html). And you may also want to check out the [PyTorch forums](https://discuss.pytorch.org/) (which also happen to use Discourse).

Of course, to discuss fastai, you can use our forums, and be sure to look through the [fastai docs](https://docs.fast.ai) too.

Don't worry if you're just starting out&mdash;little, if any, of those docs and forum threads will make any sense to you just now. But come back in a couple of weeks and you might be surprised by how useful you find them...
