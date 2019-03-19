# 第七课: 手写Resnets; U-net; Generative (adversarial) networks

## 综述

在这最后一节课里，我们将学到最新结构设计中最重要的一个技术：*skip connection* 。这项技术诞生于*resnet*, 也就是我们在做图片分类时一直在用的模型框架，为我们带来了顶级的表现。我们还会学习*U-net*结构，这项技术采用另一种skip connection，极大地改进了segmentation的效果（这项技术同样适用于其他任务，只要他们的输入值和输出值结构相似）。

之后我们将用U-net训练 *super-resolution* 模型。这个模型能够提升低像素图片的清晰度。我们的模型不仅可以提升像素，还能去除jpeg格式的杂物，以及消除文字水印。

为了让模型生成高质量输出，我们需要设计一款特制损失函数来将特征损失值（也就是所谓的*perceptual loss*)和*gram* 损失值做融合。这些技巧同样适用于很多其他的图片生成任务，比如图片上色。

最后，我们讲学习一个很新的损失函数，被称为*generative adversarial loss* (被用于生成对抗网络模型，也就是*GANs*），它能改进生成模型在某任务背景下的表现，但速度上会有所牺牲。

我们在本课中展示但还未正式做学术发表的技巧，如下：

- 利用迁移学习，做到比常规方法更快更可靠的训练GANs
- 在结构设计创新与损失函数使用方法的融合上，也做出了史无前例的创新

对比之前常规方法需要数天时间，我们只是训练了数小时，模型表现已经非常靓丽。

### 课程资源

- [非常详尽的第七课课程笔记](https://github.com/hiromis/notes/blob/master/Lesson7.md) - 感谢 @hiromi
- Notebooks:
  - [lesson7-resnet-mnist.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson7-resnet-mnist.ipynb)
  - [lesson7-superres-gan.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres-gan.ipynb)
  - [lesson7-superres-imagenet.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres-imagenet.ipynb)
  - [lesson7-superres.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres.ipynb)
  - [lesson7-wgan.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson7-wgan.ipynb)
  - [lesson7-human-numbers.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson7-human-numbers.ipynb)
- [Lesson 7 in-class discussion thread](https://forums.fast.ai/t/lesson-7-in-class-chat/32554/118)
- [Lesson 7 advanced discussion](https://forums.fast.ai/t/lesson-7-further-discussion/32555)

### 其他资源

- [可视化神经网络的损失值的地表风景图](https://arxiv.org/abs/1712.09913) 论文
- [ Convolution arithmetic](https://github.com/vdumoulin/conv_arithmetic) 在课程中展示的论文
- [ Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) 论文
- [Github对Jeremy的采访](https://www.youtube.com/watch?v=v16uzPYho4g)
- [ipyexperiments](https://github.com/stas00/ipyexperiments/) -  @stas 提供了比`gc.collect` 更便捷的lib帮助释放你的GPU内存
- [Documentation improvements thread](https://forums.fast.ai/t/documentation-improvements/32550) (请帮助我们一起将文档做的更好!)

---

[编辑此页面](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-1-7.md).