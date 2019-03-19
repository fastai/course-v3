# Lesson 3: Data blocks; Multi-label classification; Segmentation

[第三课 data blocks, 多标签分类，图片像素隔离](https://forums.fast.ai/t/fast-ai-v3-2019/39325/78?u=daniel)

## Overview 综述

Lots to cover today! We start lesson 3 looking at an interesting dataset: Planet's [Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space). In order to get this data in to the shape we need it for modeling, we'll use one of fastai's most powerful (and unique!) tools: the [data block API](https://docs.fast.ai/data_block.html). We'll be coming back to this API many times over the coming lessons, and mastery of it will make you a real fastai superstar! Once you've finished this lesson, if you're ready to learn more about the data block API, have a look at this great article: [Finding Data Block Nirvana](https://blog.usejournal.com/finding-data-block-nirvana-a-journey-through-the-fastai-data-block-api-c38210537fe4), by Wayde Gilliam.

本节课内容很多！一开始我们要看一个非常有趣的数据集：Planet's [Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space). 为了让数据能“喂给”模型，我们需要用fastai强大且独特的[data block API](https://docs.fast.ai/data_block.html)工具来处理数据。在后续的课时中，我们也会反复使用这个API，数量掌握它能让你成为真正的fastai超级明星！当你完成本节课，如果你准备好学习更多data block API，可以看看这篇很棒的文章[Finding Data Block Nirvana](https://blog.usejournal.com/finding-data-block-nirvana-a-journey-through-the-fastai-data-block-api-c38210537fe4), 作者是 Wayde Gilliam.

One important feature of the Planet dataset is that it is a *multi-label* dataset. That is: each satellite image can contain *multiple* labels, whereas previous datasets we've looked at have had exactly one label per image. We'll look at what changes we need to make to work with multi-label datasets.

planet数据集一个重要特征是多标签*multi-label*。也就是说：每张卫星图片可以包含多个标签/标注，而之前的数据集我们面对的是一张图对应一个标注。我们会学到需要做哪些调整来处理这个多标签问题。

Next, we will look at *image segmentation*, which is the process of labeling every pixel in an image with a category that shows what kind of object is portrayed by that pixel. We will use similar techniques to the earlier image classification models, with a few tweaks. fastai makes image segmentation modeling and interpretation just as easy as image classification, so there won't be too many tweaks required.
接下来，我们将学习*image segmentation 图片像素隔离*，也就是对图片中每一个像素做类别标注，从而知道哪个像素对应哪个物体。我们会对前期所学的技巧做一些调整。fastai将图片像素隔离建模和解读做得跟图片分类一样简单，因此不会有太多需要调整的地方。

We will be using the popular Camvid dataset for this part of the lesson. In future lessons, we will come back to it and show a few extra tricks. Our final Camvid model will have dramatically lower error than an model we've been able to find in the academic literature!
我们将用著名的Camvid数据集来做图片像素隔离。后续课时中，还会回头学习更多技巧。我们最终Camvid模型对比所能找到的已发表的最优学术水平，将进一步大幅降低错误率。

What if your dependent variable is a continuous value, instead of a category? We answer that question next, looking at a [keypoint](https://stackoverflow.com/questions/29133085/what-are-keypoints-in-image-processing) dataset, and building a model that predicts face keypoints with high accuracy.
如果你的目标变量是连续的，而非类别，怎么办？我们将用下一个数据集[keypoint](https://stackoverflow.com/questions/29133085/what-are-keypoints-in-image-processing)来回答，我们将构建一个模型做高精度的脸部关键点预测。

## Resources资源

### Lesson resources 课程资源

- [第三课笔记](https://forums.fast.ai/t/deep-learning-lesson-3-notes/29829) from @PoonamV
- [第三课 详尽笔记](https://github.com/hiromis/notes/blob/master/Lesson3.md) by @hiromi
- __课程 notebooks需要 fastai 1.0.21 或更新__. 请用 `conda install -c fastai fastai` (或其他合适你平台的代码),不要忘记用 `git pull` 更新 notebooks
- Notebooks:
  - [lesson3-planet.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-planet.ipynb)
  - [lesson3-camvid.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-camvid.ipynb)
  - [lesson3-imdb.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb)
  - [lesson3-head-pose.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-head-pose.ipynb)
- [Lesson 3 in-class discussion](https://forums.fast.ai/t/lesson-3-chat/29733)
- [Links to different parts in video](https://forums.fast.ai/t/lesson-3-links-to-different-parts-in-video/30077) by @melonkernel

### Other resources 其他资源

- 介绍ML背景知识的在线课程：
-- [Introduction to Machine Learning for Coders](https://course.fast.ai/ml)  作者 @jeremy 
-- [Machine Learning](https://www.coursera.org/learn/machine-learning) 作者 Andrew Ng (coursera)
- [Video Browser with Searchable Transcripts](http://videos.fast.ai/) Password: deeplearningSF2018 (do not share outside the forum group) -  [PRs welcome.]( https://github.com/zcaceres/fastai-video-browser)
- [Quick and easy model deployment](https://course.fast.ai/deployment_render.html) using Render
- [Introduction to Kaggle API in Google Colab (Part-I)](https://mmiakashs.github.io/blog/2018-09-20-kaggle-api-google-colab/)  作者  @mmiakashs
- [Data block API](https://docs.fast.ai/data_block.html)
- [Python partials](https://docs.python.org/3/library/functools.html#functools.partial)
- [MoviePy](https://zulko.github.io/moviepy)  @rachel提到的 python 视频剪辑工具
- [WebRTC example for web video](https://github.com/etown/dl1/blob/master/face/static/index.html) 作者 @etown
- Nov 14 Meetup (wait list) [Conversation between Jeremy Howard and Leslie Smith](https://www.meetup.com/sfmachinelearning/events/255566613/)
- [List of transforms](https://docs.fast.ai/vision.transform.html#List-of-transforms) in `vision.transform` package

### Further reading 深入阅读

- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) Leslie Smith的论文
- [ULMFit fine-tuning for NLP Classification](http://nlp.fast.ai/category/classification.html) used in [`language_model_learner()`](https://docs.fast.ai/text.html)
- [Michael Nielsen's book](http://neuralnetworksanddeeplearning.com/)

---

[编辑此页面](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-1-3.md).
