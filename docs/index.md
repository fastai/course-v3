---
title: Practical Deep Learning for Coders, v3
---

## Getting started

Welcome! If you're new to all this deep learning stuff, then don't worry&mdash;we'll take you through it all step by step. We do however assume that you've been coding for at least a year, and also that (if you haven't used Python before) you'll be putting in the extra time to learn whatever Python you need as you go. (For learning Python, we've heard good things about the [DataCamp](https://www.datacamp.com/courses/intro-to-python-for-data-science) course, as well as [Learn Python the Hard Way](https://learnpythonthehardway.org/).)

You might be surprised by what you *don't* need to become a top deep learning practitioner. You need one year coding experience, a GPU and appropriate software (see below), and that's it. You don't need much data, you don't need university-level math, and you don't need a giant data center. For more on this, see our article: [What you need to do deep learning](http://www.fast.ai/2017/11/16/what-you-need/).

### Using a GPU

To do nearly everything in this course, you'll need access to a computer with an NVIDIA GPU (unfortunately other brands of GPU are not fully supported by the main deep learning libraries). However, we don't recommend you buy one; in fact, even if you already have one, we don't suggest you use it just yet! Setting up a computer takes time and energy, and you want all your energy to focus on deep learning right now. Therefore, we instead suggest you rent access to a computer that already has everything you need preinstalled and ready to go. Costs will generally be around US$0.50 to US$1.25 per hour while you're using it. Here are some great choices:

#### Ready to run options

These are the easiest to use; they've got all the software, data, and lessons preinstalled for you. Click the link for more information on each, and setup instructions. The most important thing to remember: **when you're done, shut down your server**. It's not enough to close your browser, or turn off your own PC. You have to shut the server down using the methods described in the links below. Otherwise, you'll be paying for it until you shut it down!

- [Paperspace Gradient](gradient_tutorial.md)
- [Salamander](salamander_tutorial.md)

#### Some installation required

If you're comfortable at a command line, these options are fairly easy to get started with, and may be more flexible in the long term than the options above.

- [Google Compute Platform](gcp_tutorial.md)
- [Amazon Web Services EC2](dlami_tutorial.md)

### Jupyter notebook

Once you've finished the above steps, you'll be presented with the Jupyter Notebook screen. This is where you'll be doing nearly all your work in the course, so you'll want to get very familiar with it! You'll be learning a bit about it during the course, but you should probably spend a moment to try out the notebook tutorial.

Your first task, then, is to open the notebook tutorial! To do so, click `docs` and then `dl1` in jupyter, where you'll then see all the lesson notebooks. Click `notebook_tutorial.ipynb` and follow the instructions!

When you're done, **remember to shut down your server**.

### Our forums

Got stuck? Want to know more about some topic? Your first port of call should be [forums.fast.ai](https://forums.fast.ai/). There are thousands of students and practitioners asking and answering questions there. That means that it's likely your question has already been answered! So click the little magnifying class in the top right there, and search for the information you need; for instance, if you have some error message, paste a bit of it into the search box.

The forum software we use is called [Discourse](https://www.discourse.org/about). When you first join, it will show you some tips and tricks. There is also this [handy walk-thru](https://forums.episodeinteractive.com/t/a-quick-how-to-for-discourse/48/1) provided by another Discourse forum (not affiliated with fast.ai).

### PyTorch and fastai

We teach how to train [PyTorch](https://pytorch.org/) models using the [fastai](https://docs.fast.ai) library. These two pieces of software are deeply connected&mdash;you can't become really proficient at using fastai if you don't know PyTorch well too. Therefore, you will often need to refer to the [PyTorch docs](https://pytorch.org/docs/stable/index.html). And you may also want to check out the [PyTorch forums](https://discuss.pytorch.org/) (which also happen to use Discourse).

Of course, to discuss fastai, you can use our forums, and be sure to look through the [fastai docs](https://docs.fast.ai) too.

Don't worry if you're just starting out&mdash;little, if any, of those docs and forum threads will make any sense to you just now. But come back in a couple of weeks and you might be surprised by how useful you find them...
