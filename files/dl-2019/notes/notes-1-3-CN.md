# Lesson 3: Data blocks; Multi-label classification; Segmentation

## Overview

Lots to cover today! We start lesson 3 looking at an interesting dataset: Planet's [Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space). In order to get this data in to the shape we need it for modeling, we'll use one of fastai's most powerful (and unique!) tools: the [data block API](https://docs.fast.ai/data_block.html). We'll be coming back to this API many times over the coming lessons, and mastery of it will make you a real fastai superstar! Once you've finished this lesson, if you're ready to learn more about the data block API, have a look at this great article: [Finding Data Block Nirvana](https://blog.usejournal.com/finding-data-block-nirvana-a-journey-through-the-fastai-data-block-api-c38210537fe4), by Wayde Gilliam.

One important feature of the Planet dataset is that it is a *multi-label* dataset. That is: each satellite image can contain *multiple* labels, whereas previous datasets we've looked at have had exactly one label per image. We'll look at what changes we need to make to work with multi-label datasets.

Next, we will look at *image segmentation*, which is the process of labeling every pixel in an image with a category that shows what kind of object is portrayed by that pixel. We will use similar techniques to the earlier image classification models, with a few tweaks. fastai makes image segmentation modeling and interpretation just as easy as image classification, so there won't be too many tweaks required.

We will be using the popular Camvid dataset for this part of the lesson. In future lessons, we will come back to it and show a few extra tricks. Our final Camvid model will have dramatically lower error than an model we've been able to find in the academic literature!

What if your dependent variable is a continuous value, instead of a category? We answer that question next, looking at a [keypoint](https://stackoverflow.com/questions/29133085/what-are-keypoints-in-image-processing) dataset, and building a model that predicts face keypoints with high accuracy.

## Resources

### Lesson resources

- [Lesson notes](https://forums.fast.ai/t/deep-learning-lesson-3-notes/29829) from @PoonamV
- [Detailed lesson notes](https://github.com/hiromis/notes/blob/master/Lesson3.md) by @hiromi
- __The notebooks for this lesson require fastai 1.0.21 or later__. Please `conda install -c fastai fastai` (or the equivalent for your platform), and of course remember to `git pull` to get the latest notebooks
- Notebooks:
  - [lesson3-planet.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-planet.ipynb)
  - [lesson3-camvid.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-camvid.ipynb)
  - [lesson3-imdb.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb)
  - [lesson3-head-pose.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-head-pose.ipynb)
- [Lesson 3 in-class discussion](https://forums.fast.ai/t/lesson-3-chat/29733)
- [Links to different parts in video](https://forums.fast.ai/t/lesson-3-links-to-different-parts-in-video/30077) by @melonkernel

### Other resources

- Useful online courses for ML background:
-- [Introduction to Machine Learning for Coders](https://course.fast.ai/ml) taught by @jeremy
-- [Machine Learning](https://www.coursera.org/learn/machine-learning) taught by Andrew Ng (coursera)
- [Video Browser with Searchable Transcripts](http://videos.fast.ai/) Password: deeplearningSF2018 (do not share outside the forum group) -  [PRs welcome.]( https://github.com/zcaceres/fastai-video-browser)
- [Quick and easy model deployment](https://course.fast.ai/deployment_render.html) using Render
- [Introduction to Kaggle API in Google Colab (Part-I)](https://mmiakashs.github.io/blog/2018-09-20-kaggle-api-google-colab/) tutorial by @mmiakashs
- [Data block API](https://docs.fast.ai/data_block.html)
- [Python partials](https://docs.python.org/3/library/functools.html#functools.partial)
- [MoviePy](https://zulko.github.io/moviepy) Python module for video editing mentioned by @rachel
- [WebRTC example for web video](https://github.com/etown/dl1/blob/master/face/static/index.html) from @etown
- Nov 14 Meetup (wait list) [Conversation between Jeremy Howard and Leslie Smith](https://www.meetup.com/sfmachinelearning/events/255566613/)
- [List of transforms](https://docs.fast.ai/vision.transform.html#List-of-transforms) in `vision.transform` package

### Further reading

- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) paper by Leslie Smith
- [ULMFit fine-tuning for NLP Classification](http://nlp.fast.ai/category/classification.html) used in [`language_model_learner()`](https://docs.fast.ai/text.html)
- [Michael Nielsen's book](http://neuralnetworksanddeeplearning.com/)

---

[Edit this page](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-1-3.md).
