# Lesson 2: Data cleaning and production; SGD from scratch

## Overview

We start today's lesson learning how to build your own image classification model using your own data, including topics such as:

- Image collection
- Parallel downloading
- Creating a validation set, and
- Data cleaning, using the model to help us find data problems.

I'll demonstrate all these steps as I create a model that can take on the vital task of differentiating teddy bears from grizzly bears. Once we've got our data set in order, we'll then learn how to productionize our teddy-finder, and make it available online.

We've had some great additions since this lesson was recorded, so be sure to check out:

- The *production starter kits* on the course web site, such as [this one](https://course.fast.ai/deployment_render.html) for deploying to Render.com
- The new interactive GUI in the lesson notebook for using the model to find and fix mislabeled or incorrectly-collected images.

In the second half of the lesson we'll train a simple model from scratch, creating our own *gradient descent* loop. In the process, we'll be learning lots of new jargon, so be sure you've got a good place to take notes, since we'll be referring to this new terminology throughout the course (and there will be lots more introduced in every lesson from here on).

## Resources

### Lesson resources

- Notebooks:
    - [lesson2-download.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb)
    - [lesson2-sgd.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-sgd.ipynb)
- [Detailed lesson notes](https://github.com/hiromis/notes/blob/master/Lesson2.md) - thanks to @hiromi
- [Lesson notes ](https://forums.fast.ai/t/deep-learning-lesson-2-notes/28772) - thanks to @PoonamV
- [Lesson 2 in-class discussion](https://forums.fast.ai/t/lesson-2-chat/28722)
- [Links to different parts in the video](https://forums.fast.ai/t/lesson-2-links-to-different-parts-in-video/28777) - thanks to @melonkernel

### Other resources

- [ How (and why) to create a good validation set](https://www.fast.ai/2017/11/13/validation-sets/) by @rachel
- [There's no such thing as "not a math person"](https://www.youtube.com/watch?v=q6DGVGJ1WP4) by @rachel
- [Responder](https://github.com/kennethreitz/responder) - a web app framework built on top of Starlette
- Post about an [alternative image downloader/cleaner](https://www.christianwerner.net/tech/Build-your-image-dataset-faster/) by @cwerner
- [A tool for excluding irrelevant images from Google Image Search results](https://forums.fast.ai/t/tool-for-deleting-files-on-the-google-image-search-page-before-downloading/28900) by @melonkernel
- [ Machine Learning is Fun](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721) - source of image/number GIF animation shown in lesson
- [A systematic study of the class imbalance problem in convolutional neural networks](https://arxiv.org/abs/1710.05381), mentioned by Jeremy as a way to solve imbalanced datasets.

---

[Edit this page](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-1-2.md).
