# Lesson 7: Resnets from scratch; U-net; Generative (adversarial) networks

## Overview

In the final lesson of Practical Deep Learning for Coders we'll study one of the most important techniques in modern architectures: the *skip connection*. This is most famously used in the *resnet*, which is the architecture we've used throughout this course for image classification, and appears in many cutting edge results. We'll also look at the *U-net* architecture, which uses a different type of skip connection to greatly improve segmentation results (and also for similar tasks where the output structure is similar to the input).

We'll then use the U-net architecture to train a *super-resolution* model. This is a model which can increase the resolution of a low-quality image. Our model won't only increase resolution&mdash;it will also remove jpeg artifacts, and remove unwanted text watermarks.

In order to make our model produce high quality results, we will need to create a custom loss function which incorporates *feature loss* (also known as *perceptual loss*), along with *gram loss*. These techniques can be used for many other types of image generation task, such as image colorization.

Finally, we'll learn about a recent loss function known as *generative adversarial* loss (used in generative adversarial networks, or *GANs*), which can improve the quality of generative models in some contexts, at the cost of speed.

The techniques we show in this lesson include some unpublished research that:

- Let us train GANs more quickly and reliably than standard approaches, by leveraging transfer learning
- Combines architectural innovations and loss function approaches that haven't been used in this way before.

The results are stunning, and train in just a couple of hours (compared to previous approaches that take a couple of days).

### Lesson Resources

- [Detailed lesson notes](https://github.com/hiromis/notes/blob/master/Lesson7.md) - thanks to @hiromi
- Notebooks:
  - [lesson7-resnet-mnist.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson7-resnet-mnist.ipynb)
  - [lesson7-superres-gan.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres-gan.ipynb)
  - [lesson7-superres-imagenet.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres-imagenet.ipynb)
  - [lesson7-superres.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres.ipynb)
  - [lesson7-wgan.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson7-wgan.ipynb)
  - [lesson7-human-numbers.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson7-human-numbers.ipynb)
- [Lesson 7 in-class discussion thread](https://forums.fast.ai/t/lesson-7-in-class-chat/32554/118)
- [Lesson 7 advanced discussion](https://forums.fast.ai/t/lesson-7-further-discussion/32555)

### Other Resources

- [ Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)
- [ Convolution arithmetic](https://github.com/vdumoulin/conv_arithmetic) paper shown in class
- [ Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- [Interview with Jeremy at Github](https://www.youtube.com/watch?v=v16uzPYho4g)
- [ipyexperiments](https://github.com/stas00/ipyexperiments/) - handy lib from @stas that is even better than `gc.collect` at reclaiming your GPU memory
- [Documentation improvements thread](https://forums.fast.ai/t/documentation-improvements/32550) (please help us make the docs better!)

---

[Edit this page](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-1-7.md).
