# Lesson 1: Image classification

You can click the blue arrow buttons on the left and right panes to hide them and make more room for the video. You can search the transcript using the text box at the bottom. Scroll down this page for links to many useful resources. If you have any other suggestions for links, edits, or anything else, you'll find an "edit" link at the bottom of this (and every) notes panel.

## Overview

To follow along with the lessons, you'll need to connect to a cloud GPU provider which has the fastai library installed (recommended; it should take only 5 minutes or so, and cost under $0.50/hour), or set up a computer with a suitable GPU yourself (which can take days to get working if you're not familiar with the process, so we don't recommend it). You'll also need to be familiar with the basics of the *Jupyter Notebook* environment we use for running deep learning experiments. Up to date tutorials and recommendations for these are available from the [course website](https://course.fast.ai).

The key outcome of this lesson is that we'll have trained an image classifier which can recognize pet breeds at state of the art accuracy. The key to this success is the use of *transfer learning*, which will be a key platform for much of this course. We'll also see how to analyze the model to understand its failure modes. In this case, we'll see that the places where the model is making mistakes is in the same areas that even breeding experts can make mistakes.

We'll discuss the overall approach of the course, which is somewhat unusual in being *top-down* rather than *bottom-up*. So rather than starting with theory, and only getting to practical applications later, instead we start with practical applications, and then gradually dig deeper and deeper in to them, learning the theory as needed. This approach takes more work for teachers to develop, but it's been shown to help students a lot, for example in [education research at Harvard](https://www.gse.harvard.edu/news/uk/09/01/education-bat-seven-principles-educators) by David Perkins.

We also discuss how to set the most important *hyper-parameter* when training neural networks: the *learning rate*, using Leslie Smith's fantastic *learning rate finder* method. Finally, we'll look at the important but rarely discussed topic of *labeling*, and learn about some of the features that fastai provides for allowing you to easily add labels to your images.

If you want to more deeply understand how PyTorch really works, you may want to check out [this official PyTorch tutorial](https://pytorch.org/tutorials/beginner/nn_tutorial.html) by Jeremy&mdash;although we'd only suggest doing that once you've completed a few lessons.

## Links

### Lesson resources

- [Course site](https://course.fast.ai), including setup guides for each platform
- [Course repo](https://github.com/fastai/course-v3)
- [fastai docs](http://docs.fast.ai)
- [fastai datasets](https://course.fast.ai/datasets)
- Notebooks:
  - [00_notebook_tutorial.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/00_notebook_tutorial.ipynb)
  - [lesson1-pets.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb)
- [Detailed lesson notes](https://github.com/hiromis/notes/blob/master/Lesson1.md) - thanks to @hiromi
- [Lesson notes](https://forums.fast.ai/t/deep-learning-lesson-1-notes/27748) - thanks to @PoonamV (wiki thread - please help contribute!)
- [Lesson discussion thread](https://forums.fast.ai/t/lesson-1-discussion/27332)

### Other resources

- [Thread on creating your own image dataset](https://forums.fast.ai/t/tips-for-building-large-image-datasets/26688)
- [What you need to do deep learning](http://www.fast.ai/2017/11/16/what-you-need/) (fast.ai blog post including some basics on what GPUs are and why they're needed)
- [Original Paper for Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf)
- [The Oxford-IIIT Pet Dataset ](http://www.robots.ox.ac.uk/~vgg/data/pets/)
- [What the Regular Expressions in the notebook meant](https://medium.com/@youknowjamest/parsing-file-names-using-regular-expressions-3e85d64deb69)
- [Understanding Regular Expressions](https://youtu.be/DRR9fOXkfRE) (12 minute video)
- [Visualize Regular Expressions](https://regexr.com/)
- [Interactive tutorial to learn Regular Expressions](https://regexone.com)
- [Beginners Tutorial of Regular Expression](https://www.analyticsvidhya.com/blog/2015/06/regular-expression-python/)
- [One-Cycle Policy Fitting paper](https://arxiv.org/abs/1803.09820)
- [Visualizing and Understanding Convolutional Networks (paper)](https://arxiv.org/abs/1311.2901)

### How to scrape images

- [Official course tutorial](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb)
- https://forums.fast.ai/t/tips-for-building-large-image-datasets/26688
- https://forums.fast.ai/t/generating-image-datasets-quickly/19079
- https://forums.fast.ai/t/how-to-scrape-the-web-for-images/7446/8

---

## Edit this page

To edit this page, [click here](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-1-1.md). This will take you to a edit window at GitHub where you can submit your suggested changes. They will automatically be turned in to a [pull request](https://help.github.com/articles/about-pull-requests/) which will be reviewed by an admin prior to publication.
