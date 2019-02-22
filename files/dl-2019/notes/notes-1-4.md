# Lesson 4: NLP; Tabular data; Collaborative filtering; Embeddings

## Overview

In lesson 4 we'll dive in to *natural language processing* (NLP), using the IMDb movie review dataset. In this task, our goal is to predict whether a movie review is positive or negative; this is called *sentiment analysis*. We'll be using the [ULMFiT](https://arxiv.org/abs/1801.06146) algorithm, which was originally developed during the fast.ai 2018 course, and became part of a revolution in NLP during 2018 which led the New York Times to declare that [new systems are starting to crack the code of natural language](https://www.nytimes.com/2018/11/18/technology/artificial-intelligence-language.html). ULMFiT is today the most accurate known sentiment analysis algorithm.

The basic steps are:

1. Create (or, preferred, download a pre-trained) *language model* trained on a large corpus such as Wikipedia (a "language model" is any model that learns to predict the next word of a sentence)
1. Fine-tune this language model using your *target corpus* (in this case, IMDb movie reviews)
1. Extract the *encoder* from this fine tuned language model, and pair it with a *classifier*. Then fine-tune this model for the final classification task (in this case, sentiment analysis).

After our journey into NLP, we'll complete our practical applications for Practical Deep Learning for Coders by covering tabular data (such as spreadsheets and database tables), and collaborative filtering (recommendation systems).

For tabular data, we'll see how to use *categorical* and *continuous* variables, and how to work with the *fastai.tabular* module to set up and train a model.

Then we'll see how collaborative filtering models can be built using similar ideas to those for tabular data, but with some special tricks to get both higher accuracy and more informative model interpretation.

This brings us to the half-way point of the course, where we have looked at how to build and interpret models in each of these key application areas:

- Computer vision
- NLP
- Tabular
- Collaborative filtering

For the second half of the course, we'll learn about *how* these models really work, and how to create them ourselves from scratch. For this lesson, we'll put together some of the key pieces we've touched on so far:

- Activations
- Parameters
- Layers (affine and non-linear)
- Loss function.

We'll be coming back to each of these in lots more detail during the remaining lessons. We'll also learn about a type of layer that is important for NLP, collaborative filtering, and tabular models: the *embedding layer*. As we'll discover, an "embedding" is simply a computational shortcut for a particular type of matrix multiplication (a multiplication by a *one-hot encoded* matrix).

## Resources

### Lesson resources

- Notebooks:
  - [lesson4-collab.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson4-collab.ipynb)
  - [lesson4-tabular.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson4-tabular.ipynb)
- Excel spreadsheets:
  - [collab_filter.xlsx](https://github.com/fastai/course-v3/blob/master/files/xl/collab_filter.xlsx)
- [Links to different parts in video](https://forums.fast.ai/t/lesson-4-links-to-different-parts-in-the-video/30338) by @melonkernel
- [Lesson notes](https://forums.fast.ai/t/deep-learning-lesson-4-notes/30983) by @PoonamV
- [Detailed lesson notes](https://github.com/hiromis/notes/blob/master/Lesson4.md) by @hiromi
- [Brief lesson notes](https://medium.com/@boy1729/deep-learning-ver3-lesson-4-8f085a1e28ca) from @boy1729
- [Lesson 4 in-class discussion](https://forums.fast.ai/t/lesson-4-in-class-discussion/30318)
- [Lesson 4 advanced discussion](https://forums.fast.ai/t/lesson-4-advanced-discussion/30319)

### Other resources

- [QCon.ai keynote on Analyzing &amp; Preventing Unconscious Bias in Machine Learning](https://www.infoq.com/presentations/unconscious-bias-machine-learning)
- [PyBay keynote with case studies of what can go wrong, and steps toward solutions](https://www.youtube.com/watch?v=WC1kPtG8Iz8&list=PLtmWHNX-gukLQlMvtRJ19s7-8MrnRV6h6)
- [Workshop on Word Embeddings, Bias in ML, Why You Don't Like Math, &amp; Why AI Needs You](https://www.youtube.com/watch?v=25nC0n9ERq4)
- [AI Ethics Resources](https://www.fast.ai/2018/09/24/ai-ethics-resources/) (includes links to experts to follow)
- [What HBR Gets Wrong About Algorithms and Bias](http://www.fast.ai/2018/08/07/hbr-bias-algorithms/)
- [When Data Science Destabilizes Democracy and Facilitates Genocide](http://www.fast.ai/2017/11/02/ethics/)
- [Vim and Ctags for fast function definition lookup](https://andrew.stwrt.ca/posts/vim-ctags/)

---

[Edit this page](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-1-4.md).
